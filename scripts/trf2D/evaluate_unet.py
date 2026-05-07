import os
import sys
import torch
from tqdm import tqdm
from einops import rearrange
from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE

def main():
    import sys
    log_file = open("/home/un212/DiSWellProject/My_Masters_Project/evaluation_unet_debug.log", "w")
    class Logger:
        def write(self, message):
            sys.__stdout__.write(message)
            log_file.write(message)
            sys.__stdout__.flush()
            log_file.flush()
        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()
    sys.stdout = Logger()
    sys.stderr = Logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    import time
    start_time = time.time()
    print(f"[{time.ctime()}] Starting main evaluation...", flush=True)
    
    import matplotlib
    matplotlib.use('Agg')
    import wandb
    import matplotlib.pyplot as plt
    import numpy as np

    PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
    DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
    DATASET_NAME = "turbulent_radiative_layer_2D"
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_unet_epoch425_vrmse0.2451.pt") 
    if len(sys.argv) > 1:
        CHECKPOINT_PATH = sys.argv[1]

    # 1. Initialize Test Dataset
    rollout_steps = 5
    dataset = WellDataset(
        well_base_path=DATASET_DIR,
        well_dataset_name=DATASET_NAME,
        well_split_name="test",
        n_steps_input=4,
        n_steps_output=rollout_steps,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    print(f"[{time.ctime()}] Dataset initialized. Elapsed: {time.time() - start_time:.2f}s", flush=True)

    F = dataset.metadata.n_fields
    
    # 2. Initialize and Load Model
    model = UNetClassic(
        dim_in = 4*F,
        dim_out = 1*F,
        n_spatial_dims = 2,
        spatial_resolution = dataset.metadata.spatial_resolution, #64 x 64
        init_features = 32,
    ).to(device)

    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"[{time.ctime()}] Model loaded. Elapsed: {time.time() - start_time:.2f}s", flush=True)

    # 3. Setup DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0
    )

    # 4. Initialize WandB
    print(f"[{time.ctime()}] Initializing WandB...", flush=True)
    wandb.init(
        project="trf2D_unet_upgrade",
        job_type="evaluation",
        config={
            "checkpoint": CHECKPOINT_PATH,
            "dataset": DATASET_NAME,
            "rollout_steps": rollout_steps,
            "model": "UNetClassic"
        }
    )
    print(f"[{time.ctime()}] WandB Initialized.", flush=True)
    # Define rollout step as index for test plots
    wandb.define_metric("rollout_step")
    wandb.define_metric("vrmse_per_step", step_metric="rollout_step")

    # 5. Evaluation Loop
    total_vrmse_1s = 0.0
    per_step_vrmse = torch.zeros(rollout_steps).to(device)
    num_batches = 0
    example_logged = False
    
    print(f"Starting evaluation on test split (Rollout Steps: {rollout_steps})...", flush=True)
    batch_start = time.time()
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device) # (B, rollout_steps, Lx, Ly, F)
            
            # Single-step Prediction
            x_input = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            fx_1 = model(x_input)
            fx_1 = rearrange(fx_1, "B (To F) Lx Ly -> B To Lx Ly F", To=1, F=F)
            
            fx_1_phys = dataset.norm.denormalize_flattened(fx_1, mode="variable")
            y_1_phys = dataset.norm.denormalize_flattened(y[:, :1], mode="variable")
            
            vrmse_1 = VRMSE.eval(fx_1_phys, y_1_phys, meta=dataset.metadata).mean().item()
            total_vrmse_1s += vrmse_1

            # Multi-step Rollout
            curr_input = x.clone()
            rollout_preds = []
            for _ in range(rollout_steps):
                inp = rearrange(curr_input, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                pred = model(inp)
                pred = rearrange(pred, "B F Lx Ly -> B 1 Lx Ly F")
                rollout_preds.append(pred)
                curr_input = torch.cat([curr_input[:, 1:], pred], dim=1)
            
            fx_roll = torch.cat(rollout_preds, dim=1)
            fx_roll_phys = dataset.norm.denormalize_flattened(fx_roll, mode="variable")
            y_roll_phys = dataset.norm.denormalize_flattened(y, mode="variable")
            
            # Calculate VRMSE per step
            # VRMSE.eval returns (B, T) or (B, T, F) depending on implementation
            # Assuming (B, T) here based on train.py usage
            v_roll = VRMSE.eval(fx_roll_phys, y_roll_phys, meta=dataset.metadata) # (B, T, F)
            per_step_vrmse += v_roll.mean(dim=(0, 2))
            
            # log one example visualization
            if not example_logged:
                # Log first sample of the first batch
                # (T, Lx, Ly, F)
                gt = y_roll_phys[0].cpu().numpy()
                pred = fx_roll_phys[0].cpu().numpy()
                
                fig, axes = plt.subplots(2, rollout_steps, figsize=(rollout_steps*3, 6))
                for t in range(rollout_steps):
                    # Plotting first field
                    axes[0, t].imshow(gt[t, :, :, 0])
                    axes[0, t].set_title(f"GT step {t+1}")
                    axes[1, t].imshow(pred[t, :, :, 0])
                    axes[1, t].set_title(f"Pred step {t+1}")
                    axes[0, t].axis('off')
                    axes[1, t].axis('off')
                
                plt.tight_layout()
                wandb.log({"rollout_visualization": wandb.Image(fig)})
                plt.close(fig)
                example_logged = True
            
            num_batches += 1
            if num_batches % 10 == 0:
                print(f"[{time.ctime()}] Processed {num_batches} batches. Avg time per batch: {(time.time() - batch_start)/num_batches:.2f}s", flush=True)

    avg_vrmse_1s = total_vrmse_1s / num_batches
    avg_per_step_vrmse = (per_step_vrmse / num_batches).cpu().numpy()
    
    print(f"\nFinal Results on TEST split:")
    print(f"Mean VRMSE (1s): {avg_vrmse_1s:.6f}")
    for i, v in enumerate(avg_per_step_vrmse):
        print(f"Mean VRMSE (step {i+1}): {v:.6f}")
        # Log each step individually to create a plot
        wandb.log({
            "rollout_step": i + 1,
            "vrmse_per_step": v
        })

    wandb.log({
        "test_vrmse_1s": avg_vrmse_1s,
        "test_mean_vrmse_rollout": avg_per_step_vrmse.mean()
    })
    wandb.finish()

if __name__ == "__main__":
    main()
