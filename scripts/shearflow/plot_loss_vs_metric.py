
import csv
import matplotlib.pyplot as plt
import os

def main():
    log_path = "My_Masters_Project/results/fno_shearflow_restored/inference_log_with_loss.csv"
        
    if not os.path.exists(log_path):
        print(f"Error: Could not find {log_path}")
        return

    batch_idx = []
    vrmse = []
    loss_mse = []
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_idx.append(int(row['batch_idx']))
            vrmse.append(float(row['vrmse_norm']))
            loss_mse.append(float(row['mse']))
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: VRMSE (Metric)
    plt.subplot(2, 1, 1)
    plt.plot(batch_idx, vrmse, color='#1f77b4', label='Test VRMSE (Benchmark Metric)', alpha=0.7)
    plt.axhline(y=sum(vrmse)/len(vrmse), color='#1f77b4', linestyle='--', label=f'Mean VRMSE ({sum(vrmse)/len(vrmse):.3f})')
    plt.title('Inference Performance: Benchmark Metric (VRMSE)', fontsize=12)
    plt.yscale('log')
    plt.ylabel('VRMSE')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    # Plot 2: MSE (Loss)
    plt.subplot(2, 1, 2)
    plt.plot(batch_idx, loss_mse, color='#2ca02c', label='Inference MSE (Numerical Error)', alpha=0.7)
    plt.axhline(y=sum(loss_mse)/len(loss_mse), color='#2ca02c', linestyle='--', label=f'Mean MSE ({sum(loss_mse)/len(loss_mse):.4f})')
    plt.title('Inference Performance: Raw Numerical Error (MSE)', fontsize=12)
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('Batch Index')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)
    
    plt.tight_layout(pad=3.0)
    
    output_path = "My_Masters_Project/results/fno_shearflow_restored/loss_vs_metric.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
