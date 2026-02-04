import h5py
import numpy as np
import os

def load_split_data(hdf5_path, test_ratio=0.2, seed=42):
    """
    Loads data from a single HDF5 file and splits it into train/test sets
    along the first dimension (simulations).
    
    Args:
        hdf5_path (str): Path to the .hdf5 file.
        test_ratio (float): Fraction of simulations to use for testing.
        seed (int): Random seed for shuffling.
        
    Returns:
        tuple: (train_data, test_data)
               Each is a dictionary containing found fields (e.g. 'velocity', 'pressure', 'density').
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"File not found: {hdf5_path}")

    print(f"Loading data from {hdf5_path}...")
    
    with h5py.File(hdf5_path, 'r') as f:

        n_sims = None
    
        groups_to_check = ['t0_fields', 't1_fields', 't2_fields']
        for gname in groups_to_check:
            if gname in f:
                for key in f[gname].keys():
                    if hasattr(f[gname][key], 'shape') and len(f[gname][key].shape) > 0:
                        n_sims = f[gname][key].shape[0]
                        break
                if n_sims is not None:
                    break
        
        if n_sims is None:
             raise ValueError("Could not determine number of simulations (no valid fields found in t0/t1/t2 groups)")
        
        print(f"Found {n_sims} simulations in file.")
        
        n_test = int(n_sims * test_ratio)
        n_train = n_sims - n_test
        
        print(f"Splitting: {n_train} Train, {n_test} Test")
        
        train_data = {}
        test_data = {}
        
        for group_name in groups_to_check:
            if group_name in f:
                group = f[group_name]
                for key in group.keys():
                    ds = group[key]
                    
                    if hasattr(ds, 'shape') and len(ds.shape) > 0 and ds.shape[0] == n_sims:
                        print(f"Loading {key} from {group_name}...")
                        train_data[key] = ds[:n_train]
                        test_data[key]  = ds[n_train:]
                    else:
                        print(f"Skipping {key} in {group_name} (shape {ds.shape if hasattr(ds, 'shape') else 'unknown'} does not match n_sims={n_sims})")

    return train_data, test_data
