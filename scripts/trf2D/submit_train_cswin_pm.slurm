#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=trf2D_cswin_pm
#SBATCH --output=/home/un212/rds/hpc-work/logs/trf2D_cswin_pm_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/trf2D_cswin_pm_%j.err

# Use the .venv python explicitly (avoids cluster default env overriding activate)
PYTHON=/home/un212/DiSWellProject/My_Masters_Project/.venv/bin/python

# PYTHONPATH must point to the package ROOT (parent of controllable_patching_striding/)
# so that Hydra can resolve 'controllable_patching_striding.models.spatial_blocks.cswin'
export PYTHONPATH=/home/un212/DiSWellProject/patch-modulator:$PYTHONPATH

cd /home/un212/DiSWellProject/patch-modulator/controllable_patching_striding

module load ffmpeg/6.0

export TMPDIR=/rds/user/un212/hpc-work/tmp
export TRITON_CACHE_DIR=/rds/user/un212/hpc-work/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/rds/user/un212/hpc-work/torchinductor_cache

mkdir -p $TMPDIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR

export CUDA_LAUNCH_BLOCKING=1
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1

$PYTHON -u train.py \
    data=TRL_2D \
    server=cambridge \
    "model/processor/space_mixing=cswin_spatial_attention" \
    model/encoder=flexivit_encoder \
    model/decoder=flexivit_decoder \
    distribution=local \
    model.hidden_dim=256 \
    model.groups=8 \
    model.processor.time_mixing.num_heads=8 \
    +model.processor.space_mixing.window_size=8 \
    model.processor_blocks=8 \
    model.drop_path=0.1 \
    model.twod_only=True \
    model.causal_in_time=False \
    model.jitter_patches=False \
    data.module_parameters.batch_size=2 \
    data.module_parameters.n_steps_input=4 \
    data.module_parameters.n_steps_output=1 \
    trainer.max_epoch=300 \
    trainer.prediction_type=delta \
    trainer.max_rollout_steps=10 \
    trainer.enable_amp=True \
    optimizer.lr=1e-3 \
    auto_resume=False \
    logger.wandb_project_name="trf2D_cswin_pm" \
    name="cswin_trf2d_hidden256_blocks8" \
    +experiment_dir=/rds/user/un212/hpc-work/experiments

