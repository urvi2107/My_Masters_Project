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
from the_well.benchmark.models import TFNO, UNetConvNext
from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.metrics.spectral import power_spectrum

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
    DATASET_DIR  = "/home/un212/DiSWellProject/My_Masters_Project/data"
    DATASET_NAME = "turbulent_radiative_layer_2D"
    RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

    FNO_CHECKPOINT     = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_fno_epoch150_vrmse0.3149.pt")
    TFNO_CHECKPOINT    = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_tfno_epoch260_vrmse0.3026.pt")
    UNET_CHECKPOINT    = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_unet_epoch500_vrmse0.2347.pt")
    CNEXT_CHECKPOINT   = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_cnextunet_epoch350_vrmse0.1988.pt")

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

    # ── Load Models ──────────────────────────────────────────────────────────
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

    print("Loading TFNO model...")
    model_tfno = TFNO(
        dim_in=4 * F,
        dim_out=1 * F,
        n_spatial_dims=2,
        spatial_resolution=dataset.metadata.spatial_resolution,
        modes1=16,
        modes2=16,
        hidden_channels=128,
    ).to(device)
    model_tfno.load_state_dict(torch.load(TFNO_CHECKPOINT, map_location=device))
    model_tfno.eval()

    print("Loading UNet model...")
    model_unet = UNetClassic(
        dim_in=4 * F,
        dim_out=1 * F,
        n_spatial_dims=2,
        spatial_resolution=spatial_shape,
        init_features=48,  # large model (66.7 MB checkpoints)
    ).to(device)
    model_unet.load_state_dict(torch.load(UNET_CHECKPOINT, map_location=device))
    model_unet.eval()

    print("Loading CNextUNet model...")
    model_cnext = UNetConvNext(
        dim_in=4 * F,
        dim_out=1 * F,
        n_spatial_dims=2,
        spatial_resolution=dataset.metadata.spatial_resolution,
        init_features=42,
        blocks_per_stage=2,
    ).to(device)
    model_cnext.load_state_dict(torch.load(CNEXT_CHECKPOINT, map_location=device))
    model_cnext.eval()

    models = {
        "FNO":       model_fno,
        "TFNO":      model_tfno,
        "U-Net":     model_unet,
        "CNext-UNet": model_cnext,
    }

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Spectral bins ─────────────────────────────────────────────────────────
    bins = torch.logspace(
        np.log10(2 * np.pi / max(spatial_shape)),
        np.log10(np.pi * np.sqrt(ndims) + 1e-6),
        4,
    ).to(device)
    bins[0] = 0.0

    # ── Accumulators ──────────────────────────────────────────────────────────
    total_samples = 0
    sum_vrmse  = {k: torch.zeros(rollout_steps, F, device=device) for k in models}
    sum_ps_res = {k: torch.zeros(rollout_steps, 3, F, device=device) for k in models}
    sum_ps_true = torch.zeros(rollout_steps, 3, F, device=device)

    print("Starting evaluation loop...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device)
            B = x.shape[0]
            total_samples += B

            # Rollout for each model
            rollout = {}
            for name, model in models.items():
                curr = x.clone()
                preds = []
                for _ in range(rollout_steps):
                    inp = rearrange(curr, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                    pred = model(inp)
                    pred = rearrange(pred, "B F Lx Ly -> B 1 Lx Ly F")
                    preds.append(pred)
                    curr = torch.cat([curr[:, 1:], pred], dim=1)
                rollout[name] = torch.cat(preds, dim=1)

            # Denormalize
            roll_phys = {k: dataset.norm.denormalize_flattened(v, mode="variable")
                         for k, v in rollout.items()}
            y_phys = dataset.norm.denormalize_flattened(y, mode="variable")

            # VRMSE
            for name in models:
                v = VRMSE.eval(roll_phys[name], y_phys, meta=dataset.metadata)  # (B, T, F)
                sum_vrmse[name] += v.sum(dim=0)

            # Spectral
            for t in range(rollout_steps):
                _, ps_true_t, _ = power_spectrum(y_phys[:, t], dataset.metadata, bins=bins)
                sum_ps_true[t] += ps_true_t.sum(dim=0)
                for name in models:
                    _, ps_res_t, _ = power_spectrum(
                        roll_phys[name][:, t] - y_phys[:, t], dataset.metadata, bins=bins
                    )
                    sum_ps_res[name][t] += ps_res_t.sum(dim=0)

    # ── Final metrics ──────────────────────────────────────────────────────────
    mean_vrmse = {k: (sum_vrmse[k] / total_samples).cpu().numpy() for k in models}
    nrmse = {
        k: torch.sqrt(sum_ps_res[k] / (sum_ps_true + 1e-7)).mean(dim=-1).cpu().numpy()
        for k in models
    }

    # Save per-model results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for k in models:
        tag = k.lower().replace("-", "").replace(" ", "")
        np.save(os.path.join(RESULTS_DIR, f"{tag}_vrmse.npy"), mean_vrmse[k])
        np.save(os.path.join(RESULTS_DIR, f"{tag}_nrmse_spectral.npy"), nrmse[k])
    print(f"Results saved to {RESULTS_DIR}")

    steps = np.arange(1, rollout_steps + 1)

    # ── Plotting ──────────────────────────────────────────────────────────────
    print("Generating plot...")

    # Field display order: P, V_y, V_x, rho  (indices 1, 3, 2, 0 in dataset)
    display_names   = ["$P$", "$V_y$", "$V_x$", r"$\rho$"]
    display_indices = [1, 3, 2, 0]

    bin_labels  = ["High Freq (Small Scales)", "Mid Freq (Intermediate Scales)", "Low Freq (Large Scales)"]
    bin_indices = [2, 1, 0]

    # Consistent colours for 4 models
    colours = {
        "FNO":        "#ff7f0e",
        "TFNO":       "#1f77b4",
        "U-Net":      "#d62728",
        "CNext-UNet": "#2ca02c",
    }
    linestyles = {
        "FNO":        "--",
        "TFNO":       "-.",
        "U-Net":      ":",
        "CNext-UNet": "-",
    }

    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(3, 2, width_ratios=[1, 1.2], figure=fig, hspace=0.08)

    # ── LEFT: One-step VRMSE bar chart ────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[:, 0])
    y_idx = np.arange(len(display_names))
    n_models = len(models)
    bar_h = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_h

    for offset, (name, _) in zip(offsets, models.items()):
        vals = mean_vrmse[name][0, display_indices]
        ax_bar.barh(y_idx + offset, vals, bar_h, label=name, color=colours[name])

    ax_bar.set_title("One-Step VRMSE", fontsize=14, fontweight="bold", pad=12)
    ax_bar.set_xlabel("VRMSE (Root Variance-Scaled MSE)", fontsize=12)
    ax_bar.set_xscale("log")
    ax_bar.set_xlim(left=0.05)
    ax_bar.set_yticks(y_idx)
    ax_bar.set_yticklabels(display_names, fontsize=12)
    ax_bar.invert_yaxis()
    ax_bar.grid(True, which="both", axis="x", linestyle=":", alpha=0.6)

    # ── RIGHT: Spectral error over rollout ────────────────────────────────────
    ax_spectral = []
    for i, b_idx in enumerate(bin_indices):
        ax = fig.add_subplot(gs[i, 1]) if i == 0 else fig.add_subplot(gs[i, 1], sharex=ax_spectral[0])
        if i == 0:
            ax.set_title("Relative Error by Frequency Bin", fontsize=14, fontweight="bold", pad=12)
        ax_spectral.append(ax)

        for name in models:
            ax.plot(steps, nrmse[name][:, b_idx],
                    color=colours[name], linestyle=linestyles[name],
                    linewidth=2.0, label=name)

        ax.set_ylabel(bin_labels[i].split(" ")[0], fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_xticks(np.arange(0, rollout_steps + 1, 10))

    # Shared legend at bottom
    handles, labels = ax_bar.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_models,
               fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1.0])

    fig_dir  = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "all_models_comparison.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Plot saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
