#!/bin/bash
# Benchmark script for turbulent_radiative_layer_2D

# Stop on error
set -e

# Ensure we are using the virtualenv
source ../.venv/bin/activate

# Set WANDB to offline to avoid login prompts
export WANDB_MODE=offline

# Navigate to the benchmark directory
# The script is in My_Masters_Project/scripts/
# We need to go to DiSWellProject/the_well/the_well/benchmark
cd ../../the_well/the_well/benchmark

# Define path to our data
# Relative to the_well/the_well/benchmark (depth 3)
# We need to go up 3 levels to DiSWellProject, then down to My_Masters_Project/data
DATA_PATH="../../../My_Masters_Project/data"
DATASET="turbulent_radiative_layer_2D"

echo "==================================================="
echo "Starting Benchmarks for $DATASET"
echo "Data Path: $DATA_PATH"
echo "==================================================="

echo ">> Run 1/4: FNO"
python train.py experiment=fno server=local data=$DATASET data.well_base_path=$DATA_PATH

echo ">> Run 2/4: TFNO"
python train.py experiment=tfno server=local data=$DATASET data.well_base_path=$DATA_PATH

echo ">> Run 3/4: U-Net (Classic)"
python train.py experiment=unet_classic server=local data=$DATASET data.well_base_path=$DATA_PATH

echo ">> Run 4/4: ConvNext U-Net"
python train.py experiment=unet_convnext server=local data=$DATASET data.well_base_path=$DATA_PATH

echo "==================================================="
echo "All Benchmarks Completed Successfully!"
echo "==================================================="
