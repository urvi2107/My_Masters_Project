import re
import matplotlib.pyplot as plt
import sys
import os

def parse_and_plot_epochs(log_file, output_file="training_metrics.png"):
    epochs = []
    train_losses = []
    val_vrmses = []
    
    # Regex for: Epoch 500: Train Loss=0.026349, Val VRMSE=0.375433
    pattern = re.compile(r"Epoch (\d+): Train Loss=([\d.]+), Val VRMSE=([\d.]+)")
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    val_vrmses.append(float(match.group(3)))
    except FileNotFoundError:
        print(f"Error: File {log_file} not found.")
        return

    if not epochs:
        print("No epoch data found in log file.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss (MSE)', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val VRMSE', color=color)
    ax2.plot(epochs, val_vrmses, color=color, label='Val VRMSE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add benchmark line if relevant (~0.44)
    ax2.axhline(y=0.44, color='gray', linestyle='--', alpha=0.5, label='Benchmark (~0.44)')

    plt.title(f"Training Progress (trf2d)\nSource: {os.path.basename(log_file)}")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    best_vrmse = min(val_vrmses)
    best_epoch = epochs[val_vrmses.index(best_vrmse)]
    print(f"Best VRMSE: {best_vrmse:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_epochs.py <path_to_log_file>")
    else:
        parse_and_plot_epochs(sys.argv[1])
