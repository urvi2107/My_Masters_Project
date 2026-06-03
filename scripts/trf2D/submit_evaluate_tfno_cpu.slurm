#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=trf2d_tfno_eval_cpu
#SBATCH --output=/home/un212/rds/hpc-work/logs/eval_cpu_tfno_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/eval_cpu_tfno_%j.err

module load ffmpeg/6.0
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

cd /home/un212/DiSWellProject/My_Masters_Project

export PYTHONPATH=$PYTHONPATH:.
python3 -u scripts/trf2D/evaluate_tfno.py
