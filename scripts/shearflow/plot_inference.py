
import csv
import matplotlib.pyplot as plt
import os
import sys

def plot_vrmse(file_path):
    print(f"Reading file: {file_path}")
    batch_indices = []
    vrmses = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            if 'vrmse' not in reader.fieldnames:
                print("Error: 'vrmse' column not found")
                return
            
            for row in reader:
                try:
                    # Skip empty values
                    if not row['vrmse'] or not row['batch_idx']:
                        continue
                        
                    v = float(row['vrmse'])
                    b = int(row['batch_idx'])
                    vrmses.append(v)
                    batch_indices.append(b)
                except ValueError:
                    continue # Skip invalid rows
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return

    if not vrmses:
        print("No valid data points found to plot")
        return
        
    print(f"Plotting {len(vrmses)} points...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(batch_indices, vrmses, marker='o', linestyle='-', alpha=0.7, markersize=3)
    plt.title('Validation RMSE over Batches')
    plt.xlabel('Batch Index')
    plt.ylabel('VRMSE')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.yscale('log')
    
    output_file = os.path.join(os.path.dirname(file_path), 'inference_vrmse_plot.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path based on user context
        csv_path = "/home/un212/DiSWellProject/My_Masters_Project/inference_log.csv"
        
    plot_vrmse(csv_path)
