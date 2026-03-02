import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from the_well.benchmark.models import FNO
from the_well.data import WellDataModule
from the_well.benchmark.metrics import VRMSE, NRMSE
from the_well.data.normalization import ZScoreNormalization
import csv
import time
import yaml
import matplotlib.pyplot as plt

def main():
    # Setup logging
    log_file = "inference_log.csv"
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_idx", "vrmse_norm", "vrmse_phys", "mse", "v_tracer", "v_pressure", "v_vx", "v_vy", "timestamp"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Data (Disable dataset normalization)
    print("Loading data...")
    datamodule = WellDataModule(
        "/home/un212/rds/hpc-work/datasets",
        "shear_flow", 
        batch_size=32, 
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=False, # Disable in DM
        normalization_type=None
    )

    # Load Normalization Stats Manually
    normalization_path = "/home/un212/rds/hpc-work/datasets/shear_flow/stats.yaml"
    with open(normalization_path, "r") as f:
        stats = yaml.safe_load(f)
    
    # Initialize Normalizer
    metadata = datamodule.train_dataset.metadata

    
    if hasattr(datamodule.train_dataset, "core_field_names"):
        # Manual override:
        # t0_fields (pressure, tracer) from metadata.field_names[0]
        # t1_fields (velocity) - manually added as 'velocity'
        t0_fields = [f for f in metadata.field_names[0] if f in stats["mean"]]
        t1_fields = ["velocity"] if "velocity" in stats["mean"] else []
        core_field_names = t0_fields + t1_fields
        # WellDataset does NOT normalize scalars (like Reynolds), only fields.
        # So core_constant_field_names should be empty (or from constant_field_names fields).
        core_constant_field_names = [] 
        print(f"Using constructed core fields: {core_field_names}")
    else:
        print("Warning: accessing core_field_names from dataset object.")
        t0_fields = [f for f in metadata.field_names[0] if f in stats["mean"]]
        t1_fields = ["velocity"] if "velocity" in stats["mean"] else []
        core_field_names = t0_fields + t1_fields
        core_constant_field_names = []



    norm = ZScoreNormalization(stats, core_field_names, core_constant_field_names)
    
    # Load Model
    print("Loading model...")
    model = FNO.from_pretrained("polymathic-ai/FNO-shear_flow")
    model = model.to(device)
    model.eval()

    # Metrics
    vrmse_metric = VRMSE()
    
    # Enable flush_denormal
    torch.set_flush_denormal(True)
    
    test_loader = datamodule.test_dataloader()
    print(f"Starting inference on {len(test_loader)} batches...")

    batch_vrmses = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # Prepare Input
            x_raw = batch["input_fields"].to(device) # (B, T, X, Y, C)
            x_raw = torch.nan_to_num(x_raw) # Handle potential NaNs in raw data
            
            # Prepare Target
            y_phys = batch["output_fields"].to(device) 
            y_phys = torch.nan_to_num(y_phys) # (B, 1, X, Y, C) expected
            if y_phys.ndim == 5 and y_phys.shape[1] == 1:
                y_phys = y_phys.squeeze(1) # (B, X, Y, C)

            # Normalize Input Manually ON DEVICE
            x_norm = norm.normalize_flattened(x_raw, "variable")
            
            # Target Normalized (for reference)
            y_norm = norm.normalize_flattened(y_phys.unsqueeze(1), "variable").squeeze(1)
            
            # Reshape: b t x y c -> b (t c) x y (TC stacking)
            # IMPORTANT: The benchmark uses (t c) which means channel changes faster than time? 
            # No, rearrangement 'b t x y c -> b (t c) x y' means time is the major index, then channel.
            x = rearrange(x_norm, "b t x y c -> b (t c) x y")

            if "constant_fields" in batch:
                c = batch["constant_fields"].to(device)
                c = torch.nan_to_num(c)
                c = norm.normalize_flattened(c, "constant")
                c = rearrange(c, "b x y c -> b c x y")
                x = torch.cat([x, c], dim=1)

            x = x.float() 
            
            # Forward: model produces normalized delta
            y_delta_norm = model(x)
            y_delta_norm = rearrange(y_delta_norm, "b c x y -> b x y c")
            
            # Get last input step in normalized space
            y_prev_norm = x_norm[:, -1, ...] # (B, X, Y, C)
            
            # Delta Prediction Logic: y_next = y_curr + delta
            y_pred_norm = y_prev_norm + y_delta_norm
            
            # Denormalize to Physical space
            y_pred_phys = norm.denormalize_flattened(y_pred_norm, "variable")
            
            # Metrics
            # Normalized VRMSE (Note: we use NRMSE directly because library VRMSE ignores eps)
            score_norm = NRMSE.eval(y_pred_norm, y_norm, metadata, eps=1e-2, norm_mode="std")
            scalar_vrmse_norm = score_norm.mean().item()
            
            # Physical VRMSE
            score_phys = NRMSE.eval(y_pred_phys, y_phys, metadata, eps=1e-2, norm_mode="std") # (B, C)
            scalar_vrmse_phys = score_phys.mean().item()
            
            # Raw Loss (MSE on normalized data)
            scalar_mse = torch.mean((y_pred_norm - y_norm)**2).item()
            
            # Per-field VRMSE (averaged over batch)
            # Assuming order: tracer, pressure, vx, vy
            per_field_vrmse = score_phys.mean(dim=0)
            v_tracer = per_field_vrmse[0].item()
            v_pressure = per_field_vrmse[1].item()
            v_vx = per_field_vrmse[2].item()
            v_vy = per_field_vrmse[3].item()

            batch_vrmses.append(scalar_vrmse_norm)
            
            # Log to CSV
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([i, scalar_vrmse_norm, scalar_vrmse_phys, scalar_mse, v_tracer, v_pressure, v_vx, v_vy, time.time()])
            
            # Visualization every 100 batches
            if i % 100 == 0:
                plt.figure(figsize=(15, 5))
                # Plot tracer (channel 0)
                plt.subplot(1, 4, 1)
                plt.imshow(y_phys[0, :, :, 0].cpu().numpy(), cmap='jet')
                plt.title("Target (Tracer)")
                plt.colorbar()
                
                plt.subplot(1, 4, 2)
                plt.imshow(y_pred_phys[0, :, :, 0].cpu().numpy(), cmap='jet')
                plt.title("Pred (Tracer)")
                plt.colorbar()
                
                plt.subplot(1, 4, 3)
                diff = (y_phys[0, :, :, 0] - y_pred_phys[0, :, :, 0]).cpu().numpy()
                plt.imshow(diff, cmap='RdBu')
                plt.title("Diff")
                plt.colorbar()

                plt.subplot(1, 4, 4)
                # Plot Delta predicted by model
                plt.imshow(y_delta_norm[0, :, :, 0].cpu().numpy(), cmap='jet')
                plt.title("Delta Norm (Tracer)")
                plt.colorbar()
                
                plot_path = f"inference_sample_batch_{i}.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"Batch {i}: Norm VRMSE={scalar_vrmse_norm:.4f}, Phys VRMSE={scalar_vrmse_phys:.4f}")
            
    # Final Results
    if len(batch_vrmses) > 0:
        mean_vrmse = np.mean(batch_vrmses)
        print(f"Inference Complete. Mean VRMSE: {mean_vrmse:.4f}")
        
        with open("final_metrics.txt", "w") as f:
            f.write(f"Mean VRMSE: {mean_vrmse:.4f}\n")

if __name__ == "__main__":
    main()
