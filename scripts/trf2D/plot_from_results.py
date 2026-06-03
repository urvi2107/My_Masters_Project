"""
plot_from_results.py
--------------------
Generates the comparison plot directly from saved .npy result files,
without reloading models or running inference.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "figures")

rollout_steps = 90
steps = np.arange(1, rollout_steps + 1)

# Map model display name -> file tag
MODEL_TAGS = {
    "FNO":        "fno",
    "TFNO":       "tfno",
    "U-Net":      "unet",
    "CNext-UNet": "cnextunet",
}

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

# Load available results (skip models whose files aren't ready yet)
mean_vrmse = {}
nrmse      = {}

for name, tag in MODEL_TAGS.items():
    vrmse_path   = os.path.join(RESULTS_DIR, f"{tag}_vrmse.npy")
    nrmse_path   = os.path.join(RESULTS_DIR, f"{tag}_nrmse_spectral.npy")
    if os.path.exists(vrmse_path) and os.path.exists(nrmse_path):
        mean_vrmse[name] = np.load(vrmse_path)   # (T, F)
        nrmse[name]      = np.load(nrmse_path)   # (T, bins)
        print(f"  Loaded {name}: vrmse shape={mean_vrmse[name].shape}, nrmse shape={nrmse[name].shape}")
    else:
        print(f"  Skipping {name} — results not found yet")

if not mean_vrmse:
    print("No results found. Run evaluations first.")
    exit(1)

models_available = list(mean_vrmse.keys())
print(f"\nPlotting with: {models_available}")

# Field display order: P, V_y, V_x, rho
display_names   = ["$P$", "$V_y$", "$V_x$", r"$\rho$"]
display_indices = [1, 3, 2, 0]

bin_labels  = ["High Freq (Small Scales)", "Mid Freq (Intermediate Scales)", "Low Freq (Large Scales)"]
bin_indices = [2, 1, 0]

fig = plt.figure(figsize=(18, 8))
gs  = GridSpec(3, 2, width_ratios=[1, 1.2], figure=fig, hspace=0.08)

# ── LEFT: One-step VRMSE bar chart ───────────────────────────────────────────
ax_bar   = fig.add_subplot(gs[:, 0])
y_idx    = np.arange(len(display_names))
n_models = len(models_available)
bar_h    = 0.18
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_h

for offset, name in zip(offsets, models_available):
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

# ── RIGHT: Spectral error over rollout ────────────────────────────────────────
ax_spectral = []
for i, b_idx in enumerate(bin_indices):
    ax = fig.add_subplot(gs[i, 1]) if i == 0 else fig.add_subplot(gs[i, 1], sharex=ax_spectral[0])
    if i == 0:
        ax.set_title("Relative Error by Frequency Bin", fontsize=14, fontweight="bold", pad=12)
    ax_spectral.append(ax)

    for name in models_available:
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

# Legend
handles, labels = ax_bar.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=n_models,
           fontsize=11, bbox_to_anchor=(0.5, 0.01))

missing = [n for n in MODEL_TAGS if n not in models_available]
if missing:
    fig.text(0.5, 0.96, f"Note: {', '.join(missing)} results not yet available",
             ha="center", fontsize=10, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.06, 1, 1.0])

os.makedirs(FIGURES_DIR, exist_ok=True)
fig_path = os.path.join(FIGURES_DIR, "all_models_comparison.png")
plt.savefig(fig_path, dpi=300)
print(f"\nPlot saved to: {fig_path}")
plt.close()
