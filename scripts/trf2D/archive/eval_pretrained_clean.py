import os
import torch
from tqdm import tqdm
from einops import rearrange
from the_well.benchmark.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.benchmark.metrics import VRMSE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
    DATASET_NAME = "turbulent_radiative_layer_2D"
    HF_MODEL_ID = "polymathic-ai/FNO-turbulent_radiative_layer_2D"

    # 1. Initialize Dataset
    # Benchmark uses context length 4, single-step output.
    dataset = WellDataset(
        well_base_path=DATASET_DIR,
        well_dataset_name=DATASET_NAME,
        well_split_name="valid",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )

    # 2. Load Model
    print(f"Loading pre-trained model: {HF_MODEL_ID}")
    model = FNO.from_pretrained(HF_MODEL_ID).to(device)
    model.eval()

    # 3. Setup DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )

    # 4. Evaluation Loop
    total_vrmse = 0.0
    num_batches = 0
    
    # We use VRMSE.eval which handles the metric calculation. 
    # It expects denormalized inputs if we want to match benchmark values exactly.
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Input stacking logic: (B, Ti, Lx, Ly, F) -> (B, Ti*F, Lx, Ly)
            x = batch["input_fields"].to(device)
            x_input = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            
            # Ground truth: (B, To, Lx, Ly, F)
            y_ref = batch["output_fields"].to(device)
            
            # Prediction
            fx = model(x_input)
            
            # Reshape prediction to match ground truth format
            # F=4 for turbulent_radiative_layer_2D
            fx = rearrange(fx, "B (To F) Lx Ly -> B To Lx Ly F", To=1, F=4)
            
            # Denormalize both prediction and reference for VRMSE calculation
            fx_denorm = dataset.norm.denormalize_flattened(fx, mode="variable")
            y_denorm = dataset.norm.denormalize_flattened(y_ref, mode="variable")
            
            # Calculate VRMSE
            # VRMSE.eval(pred, target, metadata)
            # Returns a tensor of shape (F,) or scalar mean? 
            # the_well implementation usually returns channel-wise metrics.
            batch_vrmse = VRMSE.eval(fx_denorm, y_denorm, dataset.metadata)
            
            total_vrmse += batch_vrmse.mean().item()
            num_batches += 1

    final_vrmse = total_vrmse / num_batches
    print(f"\nFinal Mean VRMSE for {HF_MODEL_ID}: {final_vrmse:.6f}")

if __name__ == "__main__":
    main()
