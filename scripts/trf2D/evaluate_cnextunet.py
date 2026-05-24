import os
import sys
import torch
from tqdm import tqdm
from einops import rearrange
from the_well.benchmark.models import UNetConvNext
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.metrics.spectral import power_spectrum

def main():
    log_file = open("/home/un212/DiSWellProject/My_Masters_Project/evaluation_cnextunet_debug.log", "w")
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
    import traceback
    import matplotlib
    matplotlib.use('Agg')
    import wandb
    import matplotlib.pyplot as plt
    import numpy as np

    start_time = time.time()

    PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
    DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
    DATASET_NAME = "turbulent_radiative_layer_2D"
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_cnextunet.pt")
    if len(sys.argv) > 1:
        CHECKPOINT_PATH = sys.argv[1]

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

    F = dataset.metadata.n_fields

    model = UNetConvNext(
        dim_in=4 * F,
        dim_out=1 * F,
        n_spatial_dims=2,
        spatial_resolution=dataset.metadata.spatial_resolution,
        init_features=42,
        blocks_per_stage=2,
    ).to(device)

    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    wandb.init(
        project="trf2D_cnextunet",
        job_type="evaluation",
        config={"checkpoint": CHECKPOINT_PATH, "dataset": DATASET_NAME, "rollout_steps": rollout_steps, "model": "UNetConvNext"},
    )
    wandb.define_metric("rollout_step")
    wandb.define_metric("vrmse_per_step", step_metric="rollout_step")

    total_vrmse_1s = 0.0
    per_step_vrmse = torch.zeros(rollout_steps).to(device)
    per_step_vrmse_fields = torch.zeros(rollout_steps, F, device=device)

    spatial_shape = dataset.metadata.spatial_resolution
    ndims = dataset.metadata.n_spatial_dims
    bins = torch.logspace(
        np.log10(2 * np.pi / max(spatial_shape)),
        np.log10(np.pi * np.sqrt(ndims) + 1e-6),
        4,
    ).to(device)
    bins[0] = 0.0

    sum_ps_res = torch.zeros(rollout_steps, 3, F, device=device)
    sum_ps_true = torch.zeros(rollout_steps, 3, F, device=device)

    num_batches = 0
    example_logged = False

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_fields"].to(device)
            y = batch["output_fields"].to(device)

            x_input = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            fx_1 = model(x_input)
            fx_1 = rearrange(fx_1, "B (To F) Lx Ly -> B To Lx Ly F", To=1, F=F)
            fx_1_phys = dataset.norm.denormalize_flattened(fx_1, mode="variable")
            y_1_phys = dataset.norm.denormalize_flattened(y[:, :1], mode="variable")
            vrmse_1 = VRMSE.eval(fx_1_phys, y_1_phys, meta=dataset.metadata).mean().item()
            total_vrmse_1s += vrmse_1

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

            v_roll = VRMSE.eval(fx_roll_phys, y_roll_phys, meta=dataset.metadata)
            per_step_vrmse += v_roll.mean(dim=(0, 2))
            per_step_vrmse_fields += v_roll.mean(dim=0)

            for t in range(rollout_steps):
                _, ps_res_t, _ = power_spectrum(fx_roll_phys[:, t] - y_roll_phys[:, t], dataset.metadata, bins=bins)
                _, ps_true_t, _ = power_spectrum(y_roll_phys[:, t], dataset.metadata, bins=bins)
                sum_ps_res[t] += ps_res_t.sum(dim=0)
                sum_ps_true[t] += ps_true_t.sum(dim=0)

            if not example_logged:
                try:
                    from the_well.benchmark.metrics.plottable_data import make_video
                    print("Generating CNextU-Net rollout video...")
                    video_out_dir = os.path.join(PROJECT_ROOT, "results")
                    make_video(fx_roll_phys[0], y_roll_phys[0], dataset.metadata, output_dir=video_out_dir, epoch_number="cnextunet")
                    video_path = os.path.join(video_out_dir, dataset.metadata.dataset_name, "rollout_video", f"epochcnextunet_{dataset.metadata.dataset_name}.mp4")
                    print(f"Rollout video saved to {video_path}")
                    if os.path.exists(video_path):
                        wandb.log({"rollout_video": wandb.Video(video_path, fps=8, format="mp4")})
                        print("Rollout video logged to WandB.")
                except Exception as e:
                    print(f"Failed to generate rollout video: {e}")
                    traceback.print_exc()
                example_logged = True

            num_batches += 1

    avg_vrmse_1s = total_vrmse_1s / num_batches
    avg_per_step_vrmse = (per_step_vrmse / num_batches).cpu().numpy()
    avg_per_step_vrmse_fields = (per_step_vrmse_fields / num_batches).cpu().numpy()
    nrmse_spectral = torch.sqrt(sum_ps_res / (sum_ps_true + 1e-7)).mean(dim=-1).cpu().numpy()

    print(f"\nFinal Results on TEST split:")
    print(f"Mean VRMSE (1s): {avg_vrmse_1s:.6f}")
    for i, v in enumerate(avg_per_step_vrmse):
        print(f"Mean VRMSE (step {i+1}): {v:.6f}")
        wandb.log({"rollout_step": i + 1, "vrmse_per_step": v})

    wandb.log({"test_vrmse_1s": avg_vrmse_1s, "test_mean_vrmse_rollout": avg_per_step_vrmse.mean()})

    os.makedirs("results", exist_ok=True)
    np.save("results/cnextunet_vrmse.npy", avg_per_step_vrmse_fields)
    np.save("results/cnextunet_nrmse_spectral.npy", nrmse_spectral)
    wandb.finish()

if __name__ == "__main__":
    main()
