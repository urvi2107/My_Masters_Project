#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=trf2bench
#SBATCH --output=/home/un212/rds/hpc-work/logs/benchmark_fno_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/benchmark_fno_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to benchmark directory
cd /home/un212/DiSWellProject/the_well/the_well/benchmark

# Run the benchmark script
# Note: trainer.epochs is likely 'epochs' in the config, checking trainer/defaults.yaml it is 'epochs'
export WANDB_MODE=disabled
python3 -u train.py experiment=fno server=user_cluster data=turbulent_radiative_layer_2D trainer.epochs=500
