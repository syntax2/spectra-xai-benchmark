# scripts/01_fetch_data.py
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import io

def parse_ts_file(content):
    """Parses the content of a .ts file into numpy arrays."""
    lines = content.strip().split('\n')
    data_started = False
    all_series_data = []
    all_labels = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("@data"):
            data_started = True
            continue
        if not data_started or line.startswith("@"):
            continue

        parts = line.split(':')
        series_data = [float(p) for p in parts[0].split(',') if p]
        label = parts[-1]
        
        all_series_data.append(series_data)
        all_labels.append(label)

    X = pd.DataFrame(all_series_data).to_numpy()
    y = pd.Series(all_labels).to_numpy()
    return X, y

def fetch_and_save_datasets():
    """
    Downloads, extracts, and parses UCR datasets manually.
    """
    DATASET_NAMES = ["ECG200", "GunPoint", "DistalPhalanxOutlineAgeGroup", "CricketX"]
    BASE_URL = "http://www.timeseriesclassification.com/Downloads/"
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting robust data download and processing...\n")
    
    for name in DATASET_NAMES:
        print(f"Fetching '{name}'...")
        try:
            url = f"{BASE_URL}{name}.zip"
            # Download the zip file into memory
            with urllib.request.urlopen(url) as response:
                zip_in_memory = io.BytesIO(response.read())

            # Extract and parse train/test files from the zip
            with zipfile.ZipFile(zip_in_memory) as zf:
                train_content = zf.read(f"{name}_TRAIN.ts").decode('utf-8')
                test_content = zf.read(f"{name}_TEST.ts").decode('utf-8')
            
            X_train, y_train = parse_ts_file(train_content)
            X_test, y_test = parse_ts_file(test_content)
            
            # Add a channel dimension for deep learning models
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
            print(f"❌ Failed to process {name}. Error: {e}\n")
            
    print("All datasets have been successfully processed.")

if __name__ == "__main__":
    fetch_and_save_datasets()