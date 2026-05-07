#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=ood_fno
#SBATCH --output=/home/un212/rds/hpc-work/logs/ood_fno_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/ood_fno_%j.err

source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

cd /home/un212/DiSWellProject/My_Masters_Project

export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/evaluate_ood_fno.py
