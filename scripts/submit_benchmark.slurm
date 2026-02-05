#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=turbulent_benchmark
#SBATCH --output=/rds/user/un212/hpc-work/logs/benchmark_%j.out
#SBATCH --error=/rds/user/un212/hpc-work/logs/benchmark_%j.err

# Load any necessary modules
# module load cuda/11.8
module load ffmpeg/6.0

# Activate environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate
export WANDB_MODE=disabled

# Create logs directory
mkdir -p /rds/user/un212/hpc-work/logs
mkdir -p /rds/user/un212/hpc-work/experiments

# Navigate to benchmark directory
cd /home/un212/DiSWellProject/the_well/the_well/benchmark

# Set paths
DATA_PATH="/home/un212/DiSWellProject/My_Masters_Project/data"
DATASET="turbulent_radiative_layer_2D"

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run the training script (example for FNO) with output redirected to hpc-work
python train.py experiment=fno server=local data=$DATASET data.well_base_path=$DATA_PATH \
    hydra.run.dir=/rds/user/un212/hpc-work/experiments/fno/\${now:%Y-%m-%d}/\${now:%H-%M-%S} \
    trainer.make_rollout_videos=False \
    optimizer.lr=1e-3
