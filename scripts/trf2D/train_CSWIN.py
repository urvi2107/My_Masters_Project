import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from models.cswinmodel import CSWinModel
from tqdm import tqdm

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from utils import LinearWarmupCosineAnnealingLR
import argparse


# Configuration
from the_well.data.normalization import ZScoreNormalization 
PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
DATASET_NAME = "turbulent_radiative_layer_2D"

def main():
    parser = argparse.ArgumentParser(description="Train FNO on TRF2D")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    parser.add_argument("--scratch", action="store_true", help="Start training from scratch, ignoring checkpoints")
    args = parser.parse_args()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup DataModule
    base_path = DATASET_DIR
    
    dataset = WellDataset(
        well_base_path=base_path,
        well_dataset_name=DATASET_NAME,
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )

    F = dataset.metadata.n_fields
    
    model = CSWinModel(
        dim_in=4*F,
        dim_out=1*F,
        n_spatial_dims=2,
        spatial_resolution=dataset.metadata.spatial_resolution,
        embed_dim=64,   
        depth=4,        
        num_heads=4     
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size = args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    print("Starting training loop...")
    
    epochs = args.epochs
    best_vrmse = float('inf')
    
    # Validation Dataset (n_steps_output=rollout_steps for ground truth)
    rollout_steps = 5
    validset = WellDataset(
        well_base_path=base_path,
        well_dataset_name=DATASET_NAME,
        well_split_name="valid",
        n_steps_input=4,
        n_steps_output=rollout_steps, 
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        validset,
        shuffle=False,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
    )

    # Resume from checkpoint if exists
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    if not args.scratch:
        # Search for the best model in checkpoints directory
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_model_cswin_epoch") and f.endswith(".pt")]
        if checkpoints:
            # Sort by epoch number to get the latest
            checkpoints.sort(key=lambda x: int(x.split("epoch")[1].split("_")[0]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            print(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            start_epoch = int(latest_checkpoint.split("epoch")[1].split("_")[0])
            # Try to extract best_vrmse from filename
            try:
                best_vrmse = float(latest_checkpoint.split("vrmse")[1].replace(".pt", ""))
            except:
                pass
        else:
            # Fallback to old location for compatibility
            old_path = os.path.join(PROJECT_ROOT, "best_model_fno_epoch55_vrmse0.3925.pt")
            if os.path.exists(old_path):
                print(f"Loading old checkpoint from {old_path}")
                model.load_state_dict(torch.load(old_path, map_location=device))
                start_epoch = 55
                best_vrmse = 0.3925
    else:
        print("Starting training from scratch as requested.")

    # Setup WandB
    wandb.init(
        project="trf2D_cswin", 
        config={
            "learning_rate": args.lr,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "n_steps_input": 4,
            "embed_dim": 64,      
            "depth": 4,
            "num_heads": 4,
            "amp": True,
            "warmup_epochs": 5,
            "model": "CSWinModel"    
        }
    )

    # Define epoch as the primary step metric for plots
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # AMP Setup
    scaler = torch.amp.GradScaler('cuda')

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, 
        warmup_epochs=5, 
        max_epochs=epochs,
        warmup_start_lr=1e-4,
        eta_min=1e-5
    )
    # If resuming, we should ideally step the scheduler to the right point
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in bar:
            x = batch["input_fields"].to(device)
            x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            y = batch["output_fields"].to(device)
            y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

            # CSWin has no FFT so bfloat16 AMP is safe here
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                fx = model(x)
                mse = (fx.float() - y.float()).square().mean()

            scaler.scale(mse).backward()
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += mse.detach().item()
            bar.set_postfix(loss=mse.detach().item())
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Prepare metrics for logging
        metrics = {
            "train_loss": avg_train_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        }
        
        # Validation
        model.eval()
        total_vrmse = 0.0
        rollout_vrmse_5 = 0.0
        num_val_batches = 0
        
        # We'll do a 5-step rollout for validation
        rollout_steps = 5

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                for val_batch in valid_loader:
                    x = val_batch["input_fields"].to(device)
                    y = val_batch["output_fields"].to(device) # (B, 5, Lx, Ly, F)
                    
                    # Single-step VRMSE (using the first target step)
                    x_input = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                    fx_1 = model(x_input)
                    fx_1 = rearrange(fx_1, "B (To F) Lx Ly -> B To Lx Ly F", F=F)
                    
                    fx_1_phys = validset.norm.denormalize_flattened(fx_1, mode="variable")
                    y_1_phys = validset.norm.denormalize_flattened(y[:, :1], mode="variable")
                    
                    vrmse_1 = VRMSE.eval(fx_1_phys, y_1_phys, meta=validset.metadata).mean().item()
                    total_vrmse += vrmse_1

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
                    fx_roll_phys = validset.norm.denormalize_flattened(fx_roll, mode="variable")
                    y_roll_phys = validset.norm.denormalize_flattened(y, mode="variable")
                    
                    vrmse_roll = VRMSE.eval(fx_roll_phys, y_roll_phys, meta=validset.metadata).mean().item()
                    rollout_vrmse_5 += vrmse_roll
                    
                    num_val_batches += 1
            
            avg_vrmse = total_vrmse / num_val_batches
            avg_roll_vrmse = rollout_vrmse_5 / num_val_batches
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val VRMSE (1s)={avg_vrmse:.6f}, Rollout (5s)={avg_roll_vrmse:.6f}")
            
            # Add validation metrics to logging dictionary
            metrics.update({
                "val_vrmse_1step": avg_vrmse,
                "val_vrmse_rollout": avg_roll_vrmse
            })
            
        # Log all metrics for the epoch at once
        wandb.log(metrics)
            
        # Save Best Model
        if avg_vrmse < best_vrmse:
            best_vrmse = avg_vrmse
            checkpoint_name = f"best_model_cswin_epoch{epoch+1}_vrmse{best_vrmse:.4f}.pt"
            torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "checkpoints", checkpoint_name))
            print(f"New best model saved! VRMSE: {best_vrmse:.4f}")

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
