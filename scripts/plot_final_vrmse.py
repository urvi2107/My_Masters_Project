
import csv
import matplotlib.pyplot as plt
import os

def main():
    log_path = "My_Masters_Project/results/fno_shearflow_restored/inference_log_stabilized.csv"
    if not os.path.exists(log_path):
        log_path = "inference_log.csv"
        
    if not os.path.exists(log_path):
        print(f"Error: Could not find {log_path}")
        return

    batch_idx = []
    vrmse_norm = []
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_idx.append(int(row['batch_idx']))
            vrmse_norm.append(float(row['vrmse_norm']))
    
    if not vrmse_norm:
        print("Error: No data found in CSV.")
        return

    mean_val = sum(vrmse_norm) / len(vrmse_norm)
    
    plt.figure(figsize=(12, 6))
    plt.plot(batch_idx, vrmse_norm, color='#1f77b4', label='Normalized VRMSE', alpha=0.7, linewidth=1)
    plt.axhline(y=mean_val, color='#d62728', linestyle='--', label=f'Mean VRMSE ({mean_val:.3f})')
    
    plt.title('FNO Inference Performance - Stabilized VRMSE (eps=1e-2)', fontsize=14, pad=15)
    plt.xlabel('Batch Index', fontsize=12)
    plt.ylabel('VRMSE (Log Scale)', fontsize=12)
    plt.yscale('log')
    
    # Modern styling
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    output_path = "My_Masters_Project/results/fno_shearflow_restored/vrmse_progression.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
