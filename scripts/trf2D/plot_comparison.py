import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

# Use Agg backend for non-interactive environments
matplotlib.use('Agg')

from neuralop.models import FNO
from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.metrics.spectral import power_spectrum

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
    DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
    DATASET_NAME = "turbulent_radiative_layer_2D"
    
    FNO_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_fno_epoch140_vrmse0.3176.pt") 
    UNET_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_unet_epoch425_vrmse0.2451.pt") 

    # 1. Initialize Test Dataset
    rollout_steps = 90
    dataset = WellDataset(
        well_base_path=DATASET_DIR,
        well_dataset_name=DATASET_NAME,
        well_split_name="test",
        n_steps_input=4,
        n_steps_output=rollout_steps,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    print(f"Dataset initialized.")

    F = dataset.metadata.n_fields
    field_names = dataset.metadata.field_names
    spatial_shape = dataset.metadata.spatial_resolution
    ndims = dataset.metadata.n_spatial_dims
    print(f"Fields ({F}): {field_names}")
    print(f"Spatial shape: {spatial_shape}")

    # 2. Initialize and Load Models
    print("Loading FNO model...")
    model_fno = FNO(
        n_modes=(16, 16),
        in_channels=4 * F,
        out_channels=1 * F,
        hidden_channels=128,
        n_layers=4,
    ).to(device)
    model_fno.load_state_dict(torch.load(FNO_CHECKPOINT, map_location=device))
    model_fno.eval()

    print("Loading UNet model...")
    model_unet = UNetClassic(
        dim_in=4 * F,
        dim_out=1 * F,
        n_spatial_dims=2,
        spatial_resolution=spatial_shape,
        init_features=32,
    ).to(device)
    model_unet.load_state_dict(torch.load(UNET_CHECKPOINT, map_location=device))
    model_unet.eval()

    # 3. Setup DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )

    # Define bins for isotropic power spectrum (evenly spaced in log space)
    bins = torch.logspace(
        np.log10(2 * np.pi / max(spatial_shape)),
        np.log10(np.pi * np.sqrt(ndims) + 1e-6),
        4,
    ).to(device)
    bins[0] = 0.0

    # Accumulators for metrics
    total_samples = 0
    sum_vrmse_fno = torch.zeros(rollout_steps, F, device=device)
    sum_vrmse_unet = torch.zeros(rollout_steps, F, device=device)

    sum_ps_res_fno = torch.zeros(rollout_steps, 3, F, device=device)
    sum_ps_res_unet = torch.zeros(rollout_steps, 3, F, device=device)
    sum_ps_true = torch.zeros(rollout_steps, 3, F, device=device)

    print("Starting evaluation loop...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device) # (B, rollout_steps, Lx, Ly, F)
            B = x.shape[0]
            total_samples += B

            # --- FNO Rollout ---
            curr_input = x.clone()
            rollout_preds_fno = []
            for _ in range(rollout_steps):
                inp = rearrange(curr_input, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                pred = model_fno(inp)
                pred = rearrange(pred, "B F Lx Ly -> B 1 Lx Ly F")
                rollout_preds_fno.append(pred)
                curr_input = torch.cat([curr_input[:, 1:], pred], dim=1)
            fx_roll_fno = torch.cat(rollout_preds_fno, dim=1)

            # --- UNet Rollout ---
            curr_input = x.clone()
            rollout_preds_unet = []
            for _ in range(rollout_steps):
                inp = rearrange(curr_input, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                pred = model_unet(inp)
                pred = rearrange(pred, "B F Lx Ly -> B 1 Lx Ly F")
                rollout_preds_unet.append(pred)
                curr_input = torch.cat([curr_input[:, 1:], pred], dim=1)
            fx_roll_unet = torch.cat(rollout_preds_unet, dim=1)

            # Denormalize to physical units
            fx_roll_fno_phys = dataset.norm.denormalize_flattened(fx_roll_fno, mode="variable")
            fx_roll_unet_phys = dataset.norm.denormalize_flattened(fx_roll_unet, mode="variable")
            y_roll_phys = dataset.norm.denormalize_flattened(y, mode="variable")

            # --- Compute VRMSE ---
            vrmse_fno = VRMSE.eval(fx_roll_fno_phys, y_roll_phys, meta=dataset.metadata) # (B, T, F)
            vrmse_unet = VRMSE.eval(fx_roll_unet_phys, y_roll_phys, meta=dataset.metadata) # (B, T, F)

            sum_vrmse_fno += vrmse_fno.sum(dim=0)
            sum_vrmse_unet += vrmse_unet.sum(dim=0)

            # --- Compute Spectral Power Spectrums ---
            # Loop over rollout steps to calculate spectral metrics step-by-step
            for t in range(rollout_steps):
                _, ps_res_fno_t, _ = power_spectrum(
                    fx_roll_fno_phys[:, t] - y_roll_phys[:, t], dataset.metadata, bins=bins
                ) # (B, 3, F)
                _, ps_res_unet_t, _ = power_spectrum(
                    fx_roll_unet_phys[:, t] - y_roll_phys[:, t], dataset.metadata, bins=bins
                ) # (B, 3, F)
                _, ps_true_t, _ = power_spectrum(
                    y_roll_phys[:, t], dataset.metadata, bins=bins
                ) # (B, 3, F)

                sum_ps_res_fno[t] += ps_res_fno_t.sum(dim=0)
                sum_ps_res_unet[t] += ps_res_unet_t.sum(dim=0)
                sum_ps_true[t] += ps_true_t.sum(dim=0)

    # 4. Compute Final Metrics
    mean_vrmse_fno = (sum_vrmse_fno / total_samples).cpu().numpy() # (T, F)
    mean_vrmse_unet = (sum_vrmse_unet / total_samples).cpu().numpy() # (T, F)

    # NRMSE per bin: sqrt(ps_res / ps_true) -> averaged over fields
    nrmse_fno = torch.sqrt(sum_ps_res_fno / (sum_ps_true + 1e-7)).mean(dim=-1).cpu().numpy() # (T, 3)
    nrmse_unet = torch.sqrt(sum_ps_res_unet / (sum_ps_true + 1e-7)).mean(dim=-1).cpu().numpy() # (T, 3)

    steps = np.arange(1, rollout_steps + 1)

    # 5. Plotting
    print("Generating plot...")
    
    # Mapping to display names in screenshot order: P, V_y, V_x, \rho
    display_names = ["$P$", "$V_y$", "$V_x$", "$\\rho$"]
    # Corresponding indices in the dataset: density (0), pressure (1), velocity_x (2), velocity_y (3)
    display_indices = [1, 3, 2, 0]
    
    bin_labels = ["High Freq (Small Scales)", "Mid Freq (Intermediate Scales)", "Low Freq (Large Scales)"]
    bin_indices = [2, 1, 0]  # Map to array columns (0=Low, 1=Mid, 2=High)

    # --- Create Figure with GridSpec ---
    fig = plt.figure(figsize=(16, 8))
    # 3 rows, 2 columns. Left column gets the bar chart, right column gets stacked spectral plots.
    gs = GridSpec(3, 2, width_ratios=[1, 1.2], figure=fig)

    # ==========================================
    # LEFT PANEL: 1-Step VRMSE (Spans all 3 rows)
    # ==========================================
    ax_bar = fig.add_subplot(gs[:, 0])
    y_indices = np.arange(len(display_names))
    bar_height = 0.35

    # UNet / U-net is red, FNO is orange
    ax_bar.barh(y_indices - bar_height/2, mean_vrmse_unet[0, display_indices], bar_height, label="U-net", color="#d62728")
    ax_bar.barh(y_indices + bar_height/2, mean_vrmse_fno[0, display_indices], bar_height, label="FNO", color="#ff7f0e")

    ax_bar.set_title("One-Step VRMSE", fontsize=14, fontweight='bold', pad=12)
    ax_bar.set_xlabel("VRMSE (Root Variance-Scaled MSE)", fontsize=12)
    ax_bar.set_xscale('log')
    ax_bar.set_xlim(left=0.08) # starts around 0.1
    ax_bar.set_yticks(y_indices)
    ax_bar.set_yticklabels(display_names, fontsize=12)
    ax_bar.invert_yaxis()  # Put P at the top, \rho at the bottom
    ax_bar.grid(True, which="both", axis='x', linestyle=':', alpha=0.6)

    # ==========================================
    # RIGHT PANEL: Stacked Spectral Bins
    # ==========================================
    ax_spectral = []
    
    for i, b_idx in enumerate(bin_indices):
        # Share x-axis with the top plot so zooming/panning/ticks link perfectly
        if i == 0:
            ax = fig.add_subplot(gs[i, 1])
            ax.set_title("Relative Error by Frequency Bin", fontsize=14, fontweight='bold', pad=12)
        else:
            ax = fig.add_subplot(gs[i, 1], sharex=ax_spectral[0])
            
        ax_spectral.append(ax)

        # Plot UNet (Solid red, dotted, no marker) and FNO (Orange, dashed, no marker)
        ax.plot(steps, nrmse_unet[:, b_idx], color="#d62728", linestyle=':', 
                linewidth=2.5, label="U-net")
        ax.plot(steps, nrmse_fno[:, b_idx], color="#ff7f0e", linestyle='--', 
                linewidth=2.5, label="FNO")

        # Formatting subplots like the user snapshot
        ax.set_ylabel(bin_labels[i].split(" ")[0], fontsize=12, fontweight='bold') 
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Hide x-axis labels for the top and middle subplots
        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time Step", fontsize=12)
            if len(steps) > 15:
                ax.set_xticks(np.arange(0, len(steps) + 1, 10))
            else:
                ax.set_xticks(steps)

    # Place a single legend for the entire figure at the bottom center
    handles, labels = ax_bar.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    
    # Save plot
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "fno_vs_unet_comparison.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Plot saved successfully to: {fig_path}")
    plt.close()

if __name__ == "__main__":
    main()
