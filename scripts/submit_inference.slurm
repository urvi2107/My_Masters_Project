#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=fno_shearflow_inference
#SBATCH --output=/home/un212/rds/hpc-work/logs/inference_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/inference_%j.err

# Load modules
# module load cuda/11.8 # Uncomment if needed, but usually handled by venv or system

# Activate environment
source /rds/user/un212/hpc-work/project_env/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run inference script
python -u My_Masters_Project/scripts/run_inference_shearflow.py
