
import os
import sys
import torch
import hydra
import matplotlib.pyplot as plt
import numpy as np
from the_well.data.datamodule import WellDataModule

def verify_scales(exp_dir):
    print(f"Verifying scales for experiment: {exp_dir}")
    
    # Locate checkpoint
    possible_ckpts = [
        os.path.join(exp_dir, "checkpoints", "last.pt"),
        os.path.join(exp_dir, "checkpoints", "recent.pt"),
        os.path.join(exp_dir, "checkpoints", "best.pt"),
        os.path.join(exp_dir, "0", "checkpoints", "last.pt"),
        os.path.join(exp_dir, "0", "checkpoints", "recent.pt")
    ]
    
    ckpt_path = None
    for p in possible_ckpts:
        if os.path.exists(p):
            ckpt_path = p
            break
    
    if ckpt_path is None:
        print(f"Checkpoint not found in {exp_dir}. Checked: last.pt, recent.pt, best.pt. Cannot verify yet.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    cfg = checkpoint.get('config')
    
    if not cfg:
        print("Error: Config not found in checkpoint.")
        return

    # Setup Data
    print("Setting up data...")
    datamodule = WellDataModule(cfg.data)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    # Setup Model
    print("Setting up model...")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get Batch
    print("Running inference...")
    with torch.no_grad():
        batch = next(iter(loader))
        if isinstance(batch, dict):
             x = batch['input_fields'].to(device)
             y = batch['output_fields'].to(device)
        else:
             x = batch[0].to(device)
             y = batch[1].to(device)
             
        # Forward
        y_pred = model(x)
        
        # To Numpy
        pred = y_pred.cpu().numpy()
        gt = y.cpu().numpy()
        
    print(f"Pred Shape: {pred.shape}, GT Shape: {gt.shape}")
    
    # Visualize Sample 0
    sample_idx = 0
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Data conditioning for plotting
    # Assuming shape includes spatial dims (H, W) at end
    if pred.ndim >= 3:
        p_slice = pred[sample_idx]
        g_slice = gt[sample_idx]
        
        # If channel dim exists, take first channel
        if p_slice.ndim == 3: # (C, H, W) or (H, W, C)
             if p_slice.shape[0] < p_slice.shape[-1]: # Channel first
                 p_map = p_slice[0]
                 g_map = g_slice[0]
             else: # Channel last
                 p_map = p_slice[..., 0]
                 g_map = g_slice[..., 0]
        else:
             p_map = p_slice
             g_map = g_slice
             
        # Histograms
        ax[0].hist(p_slice.flatten(), bins=50, alpha=0.5, label='Pred', log=True)
        ax[0].hist(g_slice.flatten(), bins=50, alpha=0.5, label='GT', log=True)
        ax[0].legend()
        ax[0].set_title("Value Distribution (Log Scale)")
        
        # Heatmaps
        im1 = ax[1].imshow(p_map, cmap='viridis')
        ax[1].set_title(f"Pred (Min:{p_map.min():.2e}, Max:{p_map.max():.2e})")
        plt.colorbar(im1, ax=ax[1])
        
        im2 = ax[2].imshow(g_map, cmap='viridis')
        ax[2].set_title(f"GT (Min:{g_map.min():.2e}, Max:{g_map.max():.2e})")
        plt.colorbar(im2, ax=ax[2])
        
        save_path = "scale_comparison.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {os.path.abspath(save_path)}")
    else:
        print("Data dimensionality too low for standard heatmap visualization.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_scales.py <experiment_dir>")
        sys.exit(1)
    
    verify_scales(sys.argv[1])
