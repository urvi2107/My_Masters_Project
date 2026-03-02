import os
import torch
from einops import rearrange
from neuralop.models import FNO
from tqdm import tqdm

from the_well.benchmark.metrics import VRMSE
from the_well.benchmark.models import FNO
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization 

# Paths
DATASET_DIR = "/home/un212/DiSWellProject/My_Masters_Project/data"
DATASET_NAME = "turbulent_radiative_layer_2D"
HF_MODEL_ID = "polymathic-ai/FNO-turbulent_radiative_layer_2D"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Validation Dataset
    validset = WellDataset(
        well_base_path=DATASET_DIR,
        well_dataset_name=DATASET_NAME,
        well_split_name="valid",
        n_steps_input=4,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
    )
    F = validset.metadata.n_fields

    # 2. Setup Model using official pre-trained weights
    print(f"Loading official pre-trained model from {HF_MODEL_ID}...")
    model = FNO.from_pretrained(HF_MODEL_ID).to(device)
    model.eval()

    # 4. Run Evaluation
    print("Running evaluation on validation set...")
    valid_loader = torch.utils.data.DataLoader(
        validset,
        shuffle=False,
        batch_size=1, # One by one for detailed metrics
        num_workers=4
    )

    all_vrmses = []
    
    # Evaluate first 20 samples to get a representative average quickly
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):
            if i >= 20: break
            
            x = batch["input_fields"].to(device)
            x_input = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
            
            y = batch["output_fields"].to(device)
            
            fx = model(x_input)
            fx = rearrange(fx, "B (To F) Lx Ly -> B To Lx Ly F", F=F)
            
            # Denormalize
            fx = validset.norm.denormalize_flattened(fx, mode="variable")
            y = validset.norm.denormalize_flattened(y, mode="variable")
            
            metric = VRMSE.eval(fx, y, meta=validset.metadata)
            all_vrmses.append(metric.cpu())

    mean_vrmse = torch.stack(all_vrmses).mean(dim=0)
    print("\nPre-trained Model Evaluation (20 samples):")
    print(f"Per-channel VRMSE: {mean_vrmse}")
    print(f"Mean VRMSE: {mean_vrmse.mean().item():.4f}")

if __name__ == "__main__":
    main()
