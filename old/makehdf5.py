import pandas as pd
import numpy as np
import h5py
import os

# Folder structure
base_dir = "csvdata"
walking_dir = os.path.join(base_dir, "Walking")
jumping_dir = os.path.join(base_dir, "Jumping")

# Function to extract user and placement info
def extract_info(filename):
    filename = filename.lower().replace(".csv", "")
    user = "isaac" if "isaac" in filename else "skander" if "skander" in filename else "ahmed"
    if "hand" in filename:
        placement = "hand"
    elif "frontpocket" in filename:
        placement = "frontpocket"
    elif "backpocket" in filename:
        placement = "backpocket"
    elif "jacket" in filename:
        placement = "jacket"
    else:
        placement = "unknown"
    return user, placement

# Function to load CSVs and rename columns
def load_csvs_from_folder(folder_path, label):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)

            # Rename the columns to standard format
            rename_map = {
                "Time (s)": "time",
                "Linear Acceleration x (m/s^2)": "x",
                "Linear Acceleration y (m/s^2)": "y",
                "Linear Acceleration z (m/s^2)": "z"
            }
            df.rename(columns=rename_map, inplace=True)

            # Only keep required columns
            if not all(col in df.columns for col in ['x', 'y', 'z', 'time']):
                print(f"Skipping {filename} (missing required columns)")
                continue

            user, placement = extract_info(filename)
            df['label'] = label
            df['user'] = user
            df['placement'] = placement
            dataframes.append(df)
    return dataframes

# Load both classes
walking_data = load_csvs_from_folder(walking_dir, "walking")
jumping_data = load_csvs_from_folder(jumping_dir, "jumping")
all_data = walking_data + jumping_data

# Merge and save
if not all_data:
    print("🚨 No valid data found. Double-check your files.")
else:
    merged_df = pd.concat(all_data, ignore_index=True)

    with h5py.File("dataset.h5", "w") as hdf:
        hdf.create_dataset("raw/data", data=merged_df[["x", "y", "z"]].values)
        hdf.create_dataset("raw/time", data=merged_df["time"].values)
        hdf.create_dataset("raw/labels", data=merged_df["label"].astype('S').values)
        hdf.create_dataset("raw/user", data=merged_df["user"].astype('S').values)
        hdf.create_dataset("raw/placement", data=merged_df["placement"].astype('S').values)

    print("✅ dataset.h5 created successfully!")