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

module load ffmpeg/6.0
# Stub libva so ffmpeg can load on nodes where libva is not installed
export LD_LIBRARY_PATH=/home/un212/DiSWellProject/My_Masters_Project/lib:$LD_LIBRARY_PATH
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

cd /home/un212/DiSWellProject/My_Masters_Project

export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/evaluate_ood_fno.py
