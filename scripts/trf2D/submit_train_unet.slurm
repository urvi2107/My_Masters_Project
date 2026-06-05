#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=trf2D_unet
#SBATCH --output=/home/un212/rds/hpc-work/logs/trf2D_unet_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/trf2D_unet_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject/My_Masters_Project

# Run the script with unbuffered output
export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/train_UNET.py --epochs 300 --batch_size 64 --lr 1e-3 --scratch
