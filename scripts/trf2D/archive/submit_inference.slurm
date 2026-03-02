#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=trf2inf
#SBATCH --output=/home/un212/rds/hpc-work/logs/inference_pretrained_%j.out
#SBATCH --error=/home/un212/rds/hpc-work/logs/inference_pretrained_%j.err

# Activate virtual environment
source /home/un212/DiSWellProject/My_Masters_Project/.venv/bin/activate

# Set Hugging Face cache if needed (optional)
# export HF_HOME=/home/un212/rds/hpc-work/hf_cache

# Run the inference script
python3 -u scripts/trf2D/inference_pretrained.py
