import os
import sys
import torch
from tqdm import tqdm
from einops import rearrange
from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.metrics.spectral import power_spectrum

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
    import traceback
    import numpy as np
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
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_unet_lr1e-3_epoch495_vrmse0.2332.pt")
    if len(sys.argv) > 1:
        CHECKPOINT_PATH = sys.argv[1]

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
    print(f"[{time.ctime()}] Dataset initialized. Elapsed: {time.time() - start_time:.2f}s", flush=True)

    F = dataset.metadata.n_fields
    
    # 2. Initialize and Load Model
    model = UNetClassic(
        dim_in = 4*F,
        dim_out = 1*F,
        n_spatial_dims = 2,
        spatial_resolution = dataset.metadata.spatial_resolution,
        init_features = 48,  # large model (66.7 MB checkpoints)
    ).to(device)

    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"[{time.ctime()}] Model loaded. Elapsed: {time.time() - start_time:.2f}s", flush=True)

    batch_size = 4 if device.type == "cpu" else 64
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )

    spatial_shape = dataset.metadata.spatial_resolution
    ndims = dataset.metadata.n_spatial_dims
    bins = torch.logspace(
        np.log10(2 * np.pi / max(spatial_shape)),
        np.log10(np.pi * np.sqrt(ndims) + 1e-6),
        4,
    ).to(device)
    bins[0] = 0.0
    # N² for Plancherel normalisation
    prod_spatial_sq = float(np.prod(np.array(spatial_shape))) ** 2

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
    per_step_vrmse_fields = torch.zeros(rollout_steps, F, device=device)
    sum_ps_res = torch.zeros(rollout_steps, 3, F, device=device)
    sum_ps_true = torch.zeros(rollout_steps, 3, F, device=device)
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

            # Multi-step rollout — denorm → re-norm at each step (matches official trainer)
            curr_input = x.clone()
            rollout_preds_phys = []
            for _ in range(rollout_steps):
                inp  = rearrange(curr_input, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                pred = model(inp)
                pred = rearrange(pred, "B F Lx Ly -> B 1 Lx Ly F")
                # Denormalize to physical space for metric accumulation
                pred_phys = dataset.norm.denormalize_flattened(pred, mode="variable")
                rollout_preds_phys.append(pred_phys)
                # Re-normalize before feeding back as next input
                pred_renorm = dataset.norm.normalize_flattened(pred_phys, mode="variable")
                curr_input = torch.cat([curr_input[:, 1:], pred_renorm], dim=1)

            fx_roll_phys = torch.cat(rollout_preds_phys, dim=1)
            y_roll_phys  = dataset.norm.denormalize_flattened(y, mode="variable")
            
            # Calculate VRMSE per step
            v_roll = VRMSE.eval(fx_roll_phys, y_roll_phys, meta=dataset.metadata) # (B, T, F)
            per_step_vrmse += v_roll.mean(dim=(0, 2))
            per_step_vrmse_fields += v_roll.mean(dim=0)

            # Spectral power spectrum — accumulate total bin energy (Plancherel-correct)
            for t in range(rollout_steps):
                _, ps_res_t, _, counts = power_spectrum(fx_roll_phys[:, t] - y_roll_phys[:, t], dataset.metadata, bins=bins, return_counts=True)
                _, ps_true_t, _, _     = power_spectrum(y_roll_phys[:, t],                       dataset.metadata, bins=bins, return_counts=True)
                bin_counts = counts[:-1].unsqueeze(-1)  # (bins-1, 1)
                sum_ps_res[t]  += (ps_res_t  * bin_counts / prod_spatial_sq).sum(dim=0)
                sum_ps_true[t] += (ps_true_t * bin_counts / prod_spatial_sq).sum(dim=0)

            if not example_logged:
                try:
                    from the_well.benchmark.metrics.plottable_data import make_video
                    import the_well.benchmark.metrics.plottable_data as _pd
                    import matplotlib.animation as _anim
                    # plottable_data imports FFMpegWriter at module level, so patch its namespace
                    _OrigWriter = _pd.FFMpegWriter
                    class _Mpeg4Writer(_OrigWriter):
                        def __init__(self, *args, **kwargs):
                            kwargs['codec'] = 'mpeg4'
                            extra = [a for a in kwargs.get('extra_args', [])
                                     if a not in ('-preset', 'ultrafast', 'fast', 'medium', 'slow')]
                            kwargs['extra_args'] = extra
                            super().__init__(*args, **kwargs)
                    _pd.FFMpegWriter = _Mpeg4Writer
                    print("Generating UNet rollout video...")
                    video_out_dir = os.path.join(PROJECT_ROOT, "results")
                    make_video(fx_roll_phys[0], y_roll_phys[0], dataset.metadata, output_dir=video_out_dir, epoch_number="unet")
                    _pd.FFMpegWriter = _OrigWriter
                    video_path = os.path.join(video_out_dir, dataset.metadata.dataset_name, "rollout_video", f"epochunet_{dataset.metadata.dataset_name}.mp4")
                    print(f"Rollout video saved to {video_path}")
                    if os.path.exists(video_path):
                        wandb.log({"rollout_video": wandb.Video(video_path, fps=8, format="mp4")})
                        print("Rollout video logged to WandB.")
                except Exception as e:
                    print(f"Failed to generate rollout video: {e}")
                    traceback.print_exc()
                example_logged = True
            
            num_batches += 1
            if num_batches % 10 == 0:
                print(f"[{time.ctime()}] Processed {num_batches} batches. Avg time per batch: {(time.time() - batch_start)/num_batches:.2f}s", flush=True)

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

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "unet_vrmse.npy"), avg_per_step_vrmse_fields)
    np.save(os.path.join(results_dir, "unet_nrmse_spectral.npy"), nrmse_spectral)
    print(f"Results saved to {results_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
