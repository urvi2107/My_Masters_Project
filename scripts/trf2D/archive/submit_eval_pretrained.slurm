#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_pre
#SBATCH --output=/home/un212/rds/hpc-work/logs/eval_pretrained_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/eval_pretrained_%j.err

source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate
cd /home/un212/DiSWellProject/My_Masters_Project

python3 -u scripts/trf2D/eval_pretrained_clean.py
