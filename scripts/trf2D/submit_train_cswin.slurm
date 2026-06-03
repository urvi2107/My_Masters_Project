#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=trf2d_cswin
#SBATCH --output=/home/un212/rds/hpc-work/logs/trf2D_cswin_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/trf2D_cswin_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject/My_Masters_Project

# Reduce memory fragmentation as recommended by PyTorch OOM guidance
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run the script - resumes automatically from latest checkpoint
export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/train_CSWIN.py --epochs 500 --batch_size 4 --gradient_checkpointing --scratch
