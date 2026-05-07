"""
OOD Resolution Evaluation for FNO
----------------------------------
Tests a trained FNO at multiple spatial resolutions to measure OOD generalisation.

Approach:
  1. Load test data at native resolution
  2. For each target resolution scale:
     a. Interpolate input DOWN to target resolution
     b. Run FNO forward pass at target resolution
     c. Interpolate prediction UP to native resolution
     d. Compute VRMSE against native ground truth
  3. Report VRMSE vs resolution — lower degradation = better OOD generalisation

FNO is naturally resolution-independent (Fourier modes), so this is the
baseline that CSWin (with fixed spatial resolution) will be compared against.
"""
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb

from neuralop.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE

PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
DATASET_DIR  = "/home/un212/DiSWellProject/My_Masters_Project/data"
DATASET_NAME = "turbulent_radiative_layer_2D"
CHECKPOINT   = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_fno_epoch140_vrmse0.3176.pt")

# Resolution scales to test relative to native resolution
# 1.0 = native (in-distribution), others are OOD
RESOLUTION_SCALES = [0.25, 0.5, 1.0, 2.0]

ROLLOUT_STEPS = 5


def interpolate_spatial(tensor, scale, mode="bilinear"):
    """
    Rescale a tensor's spatial dimensions (Lx, Ly) by a scale factor.
    Input:  (B, C, Lx, Ly)
    Output: (B, C, Lx*scale, Ly*scale)
    """
    if scale == 1.0:
        return tensor
    return F.interpolate(
        tensor,
        scale_factor=scale,
        mode=mode,
        align_corners=False if mode != "nearest" else None,
        recompute_scale_factor=False,
    )


def evaluate_at_scale(model, loader, dataset, F_fields, device, scale, rollout_steps):
    """Run full evaluation (1-step + rollout) at a given resolution scale."""
    total_vrmse_1s = 0.0
    per_step_vrmse = torch.zeros(rollout_steps, device=device)
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  scale={scale:.2f}x"):
            x = batch["input_fields"].to(device)   # (B, Ti, Lx, Ly, F)
            y = batch["output_fields"].to(device)  # (B, To, Lx, Ly, F)

            B, Ti, Lx, Ly, _ = x.shape
            native_size = (Lx, Ly)

            # --- Single-step prediction ---
            x_in = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

            # Downsample input to target resolution
            x_in_scaled = interpolate_spatial(x_in, scale)

            # Model forward at target resolution
            fx_scaled = model(x_in_scaled)  # (B, F_fields, Lx*s, Ly*s)

            # Upsample prediction back to native resolution
            fx_1_native = interpolate_spatial(fx_scaled, 1.0 / scale if scale != 1.0 else 1.0)
            # Make sure it exactly matches native size (rounding artefacts)
            if fx_1_native.shape[-2:] != (Lx, Ly):
                fx_1_native = F.interpolate(fx_1_native, size=(Lx, Ly), mode="bilinear", align_corners=False)

            fx_1 = rearrange(fx_1_native, "B (To F) Lx Ly -> B To Lx Ly F", To=1, F=F_fields)

            fx_1_phys = dataset.norm.denormalize_flattened(fx_1, mode="variable")
            y_1_phys  = dataset.norm.denormalize_flattened(y[:, :1], mode="variable")
            vrmse_1   = VRMSE.eval(fx_1_phys, y_1_phys, meta=dataset.metadata).mean().item()
            total_vrmse_1s += vrmse_1

            # --- Multi-step rollout ---
            curr_input = x.clone()  # always kept at native resolution
            rollout_preds = []
            for _ in range(rollout_steps):
                inp = rearrange(curr_input, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                inp_scaled = interpolate_spatial(inp, scale)
                pred_scaled = model(inp_scaled)
                # Upsample back
                pred_native = interpolate_spatial(pred_scaled, 1.0 / scale if scale != 1.0 else 1.0)
                if pred_native.shape[-2:] != (Lx, Ly):
                    pred_native = F.interpolate(pred_native, size=(Lx, Ly), mode="bilinear", align_corners=False)
                pred = rearrange(pred_native, "B F Lx Ly -> B 1 Lx Ly F")
                rollout_preds.append(pred)
                curr_input = torch.cat([curr_input[:, 1:], pred], dim=1)

            fx_roll = torch.cat(rollout_preds, dim=1)
            fx_roll_phys = dataset.norm.denormalize_flattened(fx_roll, mode="variable")
            y_roll_phys  = dataset.norm.denormalize_flattened(y, mode="variable")

            v_roll = VRMSE.eval(fx_roll_phys, y_roll_phys, meta=dataset.metadata)
            per_step_vrmse += v_roll.mean(dim=(0, 2))

            num_batches += 1

    avg_vrmse_1s   = total_vrmse_1s / num_batches
    avg_per_step   = (per_step_vrmse / num_batches).cpu().numpy()
    avg_roll_vrmse = avg_per_step.mean()
    return avg_vrmse_1s, avg_roll_vrmse, avg_per_step


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset ---
    dataset = WellDataset(
        well_base_path=DATASET_DIR,
        well_dataset_name=DATASET_NAME,
        well_split_name="test",
        n_steps_input=4,
        n_steps_output=ROLLOUT_STEPS,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    native_res = dataset.metadata.spatial_resolution
    print(f"Native resolution: {native_res}")
    F_fields = dataset.metadata.n_fields

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- Model ---
    model = FNO(
        n_modes=(16, 16),
        in_channels=4 * F_fields,
        out_channels=1 * F_fields,
        hidden_channels=128,
        n_layers=4,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # --- WandB ---
    wandb.init(
        project="trf2D_ood",
        name="FNO_ood_resolution",
        config={
            "model": "FNO",
            "checkpoint": CHECKPOINT,
            "native_resolution": str(native_res),
            "resolution_scales": RESOLUTION_SCALES,
            "rollout_steps": ROLLOUT_STEPS,
        }
    )
    wandb.define_metric("resolution_scale")
    wandb.define_metric("*", step_metric="resolution_scale")

    # --- Evaluate at each scale ---
    results = {}
    print(f"\nTesting at scales: {RESOLUTION_SCALES}")
    print(f"Native resolution: {native_res}\n")

    for scale in RESOLUTION_SCALES:
        target_h = int(native_res[0] * scale)
        target_w = int(native_res[1] * scale)
        label = "IN-DISTRIBUTION" if scale == 1.0 else "OOD"
        print(f"[{label}] Scale {scale:.2f}x → {target_h}×{target_w}")

        vrmse_1s, vrmse_roll, per_step = evaluate_at_scale(
            model, loader, dataset, F_fields, device, scale, ROLLOUT_STEPS
        )
        results[scale] = {"vrmse_1s": vrmse_1s, "vrmse_roll": vrmse_roll, "per_step": per_step}

        print(f"  VRMSE (1-step):  {vrmse_1s:.4f}")
        print(f"  VRMSE (rollout): {vrmse_roll:.4f}")

        wandb.log({
            "resolution_scale": scale,
            "target_resolution_h": target_h,
            "vrmse_1step": vrmse_1s,
            "vrmse_rollout": vrmse_roll,
        })

    # --- Summary table ---
    print("\n" + "="*55)
    print(f"{'Scale':<10} {'Resolution':<15} {'VRMSE 1-step':<15} {'VRMSE Rollout'}")
    print("="*55)
    for scale in RESOLUTION_SCALES:
        r = results[scale]
        h = int(native_res[0] * scale)
        w = int(native_res[1] * scale)
        tag = " ← native" if scale == 1.0 else ""
        print(f"{scale:<10.2f} {h}×{w:<12} {r['vrmse_1s']:<15.4f} {r['vrmse_roll']:.4f}{tag}")
    print("="*55)

    # --- Plot VRMSE vs scale ---
    scales = RESOLUTION_SCALES
    vrmse_1s_vals   = [results[s]["vrmse_1s"]   for s in scales]
    vrmse_roll_vals = [results[s]["vrmse_roll"]  for s in scales]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, vrmse_1s_vals,   "o-", label="VRMSE 1-step")
    ax.plot(scales, vrmse_roll_vals, "s--", label="VRMSE Rollout (5-step)")
    ax.axvline(x=1.0, color="gray", linestyle=":", label="Native resolution")
    ax.set_xlabel("Resolution scale")
    ax.set_ylabel("VRMSE")
    ax.set_title("FNO — OOD Generalisation across Resolutions")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(scales)
    ax.set_xticklabels([f"{s:.2f}x" for s in scales])
    plt.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, "figures", "fno_ood_resolution.png"), dpi=150)
    wandb.log({"ood_resolution_plot": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
