#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --time=24:00:00
#SBATCH --job-name=trf2sz_500
#SBATCH --output=/home/un212/rds/hpc-work/logs/train_zscore_500_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/train_zscore_500_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject/My_Masters_Project

# Run the script with unbuffered output
python3 -u scripts/trf2D/train.py
