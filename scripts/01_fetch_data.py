# scripts/01_fetch_data.py
import numpy as np
from pathlib import Path
from sktime.datasets import load_UCR_UEA_dataset

def fetch_and_save_datasets():
    """
    Fetches the recommended datasets from the UCR archive using sktime
    and saves them locally as compressed .npz files.
    """
    DATASET_NAMES = ["ECG200", "GunPoint", "DistalPhalanxOutlineAgeGroup", "CricketX"]
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting dataset download and processing with sktime...\n")
    
    for name in DATASET_NAMES:
        print(f"Fetching '{name}'...")
        try:
            # sktime loads splits separately. We'll load train and test.
            # It returns pandas objects, which we'll convert to numpy arrays.
            X_train_pd, y_train = load_UCR_UEA_dataset(name, split="train", return_X_y=True)
            X_test_pd, y_test = load_UCR_UEA_dataset(name, split="test", return_X_y=True)

            # Convert pandas DataFrames to NumPy arrays
            # sktime data can have multiple dimensions, we select the first (dim_0)
            # and stack the series to get the correct (n_samples, n_timesteps) shape.
            X_train = np.stack(X_train_pd['dim_0'].to_numpy())
            X_test = np.stack(X_test_pd['dim_0'].to_numpy())

            # Add a channel dimension for deep learning models (n_samples, n_timesteps, 1)
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

            output_file = output_dir / f"{name}.npz"
            
            np.savez_compressed(
                output_file,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
            
            print(f"✅ Saved '{name}' to {output_file}")
            print(f"   Train shapes: X={X_train.shape}, y={y_train.shape}")
            print(f"   Test shapes:  X={X_test.shape}, y={y_test.shape}\n")

        except Exception as e:
            print(f"❌ Failed to download or process {name}. Error: {e}\n")
            
    print("All datasets have been successfully downloaded and saved.")

if __name__ == "__main__":
    fetch_and_save_datasets()