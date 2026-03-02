import sys
import os
import torch
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Setup path to include 'the_well'
# Script is in My_Masters_Project/scripts
# the_well is in DiSWellProject/the_well
sys.path.append(os.path.abspath("../../the_well"))

from the_well.benchmark.trainer.training import Trainer

def main():
    # Setup logging
    logging.basicConfig(level=logging.WARNING) # Reduce noise

    # Config and checkpoint paths
    # Relative to My_Masters_Project/scripts
    config_path = "../notebooks/experiments/turbulent_radiative_layer_2D-fno-FNO-0.01/0/extended_config.yaml"
    checkpoint_path = "../notebooks/experiments/turbulent_radiative_layer_2D-fno-FNO-0.01/0/checkpoints/best.pt"

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix base path if it was absolute in another environment, but here it looks absolute and correct.
    # /home/un212/DiSWellProject/My_Masters_Project/data/ 
    # Check if data path exists
    if not os.path.exists(cfg.data.well_base_path):
        print(f"Warning: Data path {cfg.data.well_base_path} does not exist.")
    
    print("Instantiating components...")
    datamodule = instantiate(cfg.data)
    # datamodule.setup() # Not needed for WellDataModule

    # Extract metadata for model init (Logic from train.py)
    dset_metadata = datamodule.train_dataset.metadata
    n_input_fields = (
        cfg.data.n_steps_input * dset_metadata.n_fields
        + dset_metadata.n_constant_fields
    )
    n_output_fields = dset_metadata.n_fields
    
    print(f"Model Init: n_spatial_dims={dset_metadata.n_spatial_dims}, spatial_resolution={dset_metadata.spatial_resolution}, dim_in={n_input_fields}, dim_out={n_output_fields}")

    model = instantiate(
        cfg.model,
        n_spatial_dims=dset_metadata.n_spatial_dims,
        spatial_resolution=dset_metadata.spatial_resolution,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
    )
    # Instantiate optimizer even if not used (Trainer needs it)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    loss_fn = instantiate(cfg.trainer.loss_fn)
    
    lr_scheduler = None
    if "lr_scheduler" in cfg and cfg.lr_scheduler:
         lr_scheduler = instantiate(
             cfg.lr_scheduler, 
             optimizer=optimizer,
             max_epochs=cfg.trainer.epochs,
             warmup_start_lr=cfg.optimizer.lr * 0.1,
             eta_min=cfg.optimizer.lr * 0.1,
         )

    print("Initializing Trainer and loading checkpoint...")
    # Mock folders
    trainer = Trainer(
        checkpoint_folder="/tmp",
        artifact_folder="/tmp",
        viz_folder="/tmp",
        formatter=cfg.trainer.formatter,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.trainer.epochs,
        checkpoint_frequency=cfg.trainer.checkpoint_frequency,
        val_frequency=cfg.trainer.val_frequency,
        rollout_val_frequency=cfg.trainer.rollout_val_frequency,
        max_rollout_steps=cfg.trainer.max_rollout_steps,
        short_validation_length=cfg.trainer.short_validation_length,
        make_rollout_videos=False,
        num_time_intervals=cfg.trainer.num_time_intervals,
        lr_scheduler=lr_scheduler,
        device=device,
        checkpoint_path=checkpoint_path,
        is_distributed=False
    )

    print("Running validation loops...")
    
    # Rollout Test
    print("\n--- Rollout Test Set ---")
    try:
        rollout_test_loader = datamodule.rollout_test_dataloader()
        loss, logs = trainer.validation_loop(
            rollout_test_loader, 
            valid_or_test="rollout_test", 
            full=True
        )
        for k, v in logs.items():
            if "VRMSE" in k:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Error in rollout test: {e}")

    # Standard Test
    print("\n--- Standard Test Set ---")
    try:
        test_loader = datamodule.test_dataloader()
        loss, logs = trainer.validation_loop(
            test_loader, 
            valid_or_test="test", 
            full=True
        )
        for k, v in logs.items():
            if "VRMSE" in k:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Error in standard test: {e}")

if __name__ == "__main__":
    main()
