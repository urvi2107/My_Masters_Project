#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=trf2d_eval
#SBATCH --output=/home/un212/rds/hpc-work/logs/eval_test_trf2d_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/eval_test_trf2d_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject/My_Masters_Project

# Load ffmpeg module for rollout video generation
export LD_LIBRARY_PATH=/home/un212/DiSWellProject/My_Masters_Project/lib:$LD_LIBRARY_PATH
module load ffmpeg/6.0

# Run the evaluation script
export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/evaluate.py
