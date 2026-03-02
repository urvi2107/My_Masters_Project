import os
import sys
import torch
from tqdm import tqdm
from einops import rearrange
from neuralop.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    PROJECT_ROOT = "/home/un212/DiSWellProject/My_Masters_Project"
    DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
    DATASET_NAME = "turbulent_radiative_layer_2D"
    CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model_fno.pt") 
    if len(sys.argv) > 1:
        CHECKPOINT_PATH = sys.argv[1]

    # 1. Initialize Test Dataset
    dataset = WellDataset(
        well_base_path=DATASET_DIR,
        well_dataset_name=DATASET_NAME,
        well_split_name="test",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )

    F = dataset.metadata.n_fields
    
    # 2. Initialize and Load Model
    model = FNO(
        n_modes=(16,16),
        in_channels = 4*F,
        out_channels = 1*F,
        hidden_channels = 128,
        n_layers = 4,
    ).to(device)

    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    # 3. Setup DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )

    # 4. Evaluation Loop
    total_vrmse = 0.0
    num_batches = 0
    
    print("Starting evaluation on test split...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["input_fields"].to(device)
            x_input = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            
            y_ref = batch["output_fields"].to(device)
            
            # Prediction
            fx = model(x_input)
            
            # Reshape prediction to match ground truth format
            fx = rearrange(fx, "B (To F) Lx Ly -> B To Lx Ly F", To=1, F=F)
            
            # Denormalize both prediction and reference for VRMSE calculation
            fx_denorm = dataset.norm.denormalize_flattened(fx, mode="variable")
            y_denorm = dataset.norm.denormalize_flattened(y_ref, mode="variable")
            
            # Calculate VRMSE
            batch_vrmse = VRMSE.eval(fx_denorm, y_denorm, dataset.metadata)
            
            total_vrmse += batch_vrmse.mean().item()
            num_batches += 1

    final_vrmse = total_vrmse / num_batches
    print(f"\nFinal Mean VRMSE on TEST split: {final_vrmse:.6f}")

if __name__ == "__main__":
    main()
