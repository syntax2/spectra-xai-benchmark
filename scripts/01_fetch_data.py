# scripts/01_fetch_data.py
import numpy as np
from pathlib import Path
from tsai.basics import get_UCR_data

def fetch_and_save_datasets():
    """
    Fetches the recommended datasets from the UCR archive
    and saves them locally as compressed .npz files.
    """
    # Define the datasets we finalized in the previous step
    DATASET_NAMES = ["ECG200", "GunPoint", "DistalPhalanxOutlineAgeGroup", "CricketX"]
    
    # Define the output directory (./data/), creating it if it doesn't exist
    # This directory is in our .gitignore
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting dataset download and processing...\n")
    
    for name in DATASET_NAMES:
        print(f"Fetching '{name}'...")
        
        try:
            # Use tsai to download and load the data.
            # parent_dir specifies where the raw .ts files will be cached.
            X_train, y_train, X_test, y_test = get_UCR_data(name, parent_dir=output_dir, verbose=False)
            
            # Define the final output file path
            output_file = output_dir / f"{name}.npz"
            
            # Save all arrays into a single compressed .npz file for efficient storage
            np.savez_compressed(
                output_file,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            
            print(f"✅ Saved '{name}' to {output_file}")
            print(f"   Train shapes: X={X_train.shape}, y={y_train.shape}")
            print(f"   Test shapes:  X={X_test.shape}, y={y_test.shape}\n")
        
        except Exception as e:
            print(f"❌ Failed to download or process {name}. Error: {e}\n")
            
    print("All datasets have been successfully downloaded and saved.")

if __name__ == "__main__":
    fetch_and_save_datasets()