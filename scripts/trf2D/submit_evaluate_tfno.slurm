#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=trf2d_tfno_eval
#SBATCH --output=/home/un212/rds/hpc-work/logs/eval_test_trf2d_tfno_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/eval_test_trf2d_tfno_%j.err

# Activate virtual environment
module load ffmpeg/6.0
# Stub libva so ffmpeg can load on nodes where libva is not installed
export LD_LIBRARY_PATH=/home/un212/DiSWellProject/My_Masters_Project/lib:$LD_LIBRARY_PATH
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Navigate to project root
cd /home/un212/DiSWellProject/My_Masters_Project

# Run the evaluation script
export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/evaluate_tfno.py
