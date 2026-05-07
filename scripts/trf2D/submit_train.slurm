#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --job-name=trf2D_cswin
#SBATCH --output=/home/un212/rds/hpc-work/logs/trf2D_cswin_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/trf2D_cswin_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject/My_Masters_Project

# Run the script with unbuffered output
export PYTHONPATH=$PYTHONPATH:.
# python3 -u scripts/trf2D/train.py --epochs 500 --batch_size 64 --scratch

# FIRST RUN: --scratch starts clean. Remove --scratch when resubmitting to resume from checkpoint.
python3 -u scripts/trf2D/train_CSWIN.py --epochs 300 --batch_size 32 --scratch
