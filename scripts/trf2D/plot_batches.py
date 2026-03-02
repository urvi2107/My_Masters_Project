import re
import matplotlib.pyplot as plt
import sys
import os

def parse_and_plot(log_file, output_file="loss_curve.png"):
    losses = []
    
    # Regex to capture loss=1.234 or loss=1.234e-05
    # tqdm output usually looks like: ... loss=0.203]
    loss_pattern = re.compile(r"loss=([0-9]+\.?[0-9]*(?:e[-+]?[0-9]+)?)")
    
    try:
        with open(log_file, "r") as f:
            content = f.read()
            
            # Handle potential carriage returns from tqdm by splitting on any newline-like sequence
            # or just find all matches in the raw stream
            matches = loss_pattern.findall(content)
            losses = [float(m) for m in matches]
            
    except FileNotFoundError:
        print(f"Error: File {log_file} not found.")
        return

    if not losses:
        print("No loss values found in log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss", marker='.', linestyle='None')
    plt.xlabel("Batch Iteration")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss Curve\nSource: {os.path.basename(log_file)}")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.yscale("log") # Loss often spans orders of magnitude
    
    plt.savefig(output_file)
    print(f"Loss curve saved to {output_file}")
    print(f"Parsed {len(losses)} data points. Final loss: {losses[-1]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_loss_from_log.py <path_to_log_file>")
        # Default fallback for convenience during dev
        log_path = "/home/un212/rds/hpc-work/logs/traintrf2d22686518.err"
        print(f"No file specified, defaulting to: {log_path}")
    else:
        log_path = sys.argv[1]
        
    parse_and_plot(log_path)
