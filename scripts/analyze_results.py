import os
import glob
import argparse
import matplotlib.pyplot as plt
import re
import numpy as np
import torch
import h5py

def parse_logs(log_file):
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse training loss
            # Example: [DATE][...][INFO] - Epoch 2, Batch 109/109: loss 0.111, ...
            if "loss" in line and "Epoch" in line and "Batch" in line:
                match = re.search(r"loss ([0-9.]+)", line)
                if match:
                    train_losses.append(float(match.group(1)))
            
            # Parse validation loss
            # Example: [DATE][...][INFO] - Epoch 2/2: avg validation loss 20.25
            if "avg validation loss" in line:
                match = re.search(r"loss ([0-9.]+)", line)
                if match:
                    val_losses.append(float(match.group(1)))
                    
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    
    # Plot training loss (per batch, smoothed)
    plt.plot(train_losses, alpha=0.3, label='Training Loss (Batch)')
    
    # Smooth training loss
    window = 50
    if len(train_losses) > window:
        smooth_train = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(len(smooth_train)) + window//2, smooth_train, label='Training Loss (Smoothed)')
        
    # Plot validation loss (stepwise, assuming 1 val per epoch approx)
    # We map val points to where they likely occurred in the batch sequence
    if len(val_losses) > 0 and len(train_losses) > 0:
        val_indices = np.linspace(0, len(train_losses), len(val_losses) + 1)[1:]
        plt.plot(val_indices, val_losses, 'r-o', label='Validation Loss', linewidth=2)
        
    plt.xlabel('Training Steps (Batches)')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    out_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(out_path)
    print(f"Saved loss plot to: {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze Benchmark Results')
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment directory (e.g. experiments/fno/DATE/TIME)')
    args = parser.parse_args()
    
    log_file = os.path.join(args.exp_dir, 'train.log')
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    print(f"Analyzing logs from: {log_file}")
    train_losses, val_losses = parse_logs(log_file)
    
    if not train_losses:
        print("No training data found in logs.")
        return
        
    print(f"Found {len(train_losses)} training steps and {len(val_losses)} validation steps.")
    plot_losses(train_losses, val_losses, args.exp_dir)

if __name__ == "__main__":
    main()
