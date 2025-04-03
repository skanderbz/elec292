import h5py
import pandas as pd
import numpy as np

users = ["isaac", "skander", "ahmed"]

def load_hdf5_data(hdf5_path, group_name):
    """Load data from an HDF5 file and split by user."""
    data_dict = {}
    
    with h5py.File(hdf5_path, "r") as hdf:
        for user in users:
            try:
                group = hdf[f"{group_name}/{user}"]
                data = group["data"][:]
                time = group["time"][:]
                labels = group["labels"][:].astype(str)
                placement = group["placement"][:].astype(str)
                
                df = pd.DataFrame({
                    "time": time,
                    "x": data[:, 0],
                    "y": data[:, 1],
                    "z": data[:, 2],
                    "label": labels,
                    "placement": placement,
                    "user": user
                })
                data_dict[user] = df
            except KeyError:
                print(f"User '{user}' not found in {group_name} of {hdf5_path}")
    
    return data_dict

# Load raw and pre-processed data from the ./hdf5s/ directory
raw_data = load_hdf5_data("./hdf5s/dataset.h5", "raw")
processed_data = load_hdf5_data("./hdf5s/processed_dataset.h5", "processed")

# Combine raw and preprocessed data into a single HDF5 file
with h5py.File("./hdf5s/combined_dataset.h5", "w") as hdf:
    for user in users:
        if user in raw_data:
            # Save raw data
            raw_group = hdf.create_group(f"raw/{user}")
            raw_df = raw_data[user]
            raw_group.create_dataset("data", data=raw_df[["x", "y", "z"]].values)
            raw_group.create_dataset("time", data=raw_df["time"].values)
            raw_group.create_dataset("labels", data=raw_df["label"].astype('S').values)
            raw_group.create_dataset("placement", data=raw_df["placement"].astype('S').values)
        
        if user in processed_data:
            # Save pre-processed data
            processed_group = hdf.create_group(f"pre-processed/{user}")
            processed_df = processed_data[user]
            processed_group.create_dataset("data", data=processed_df[["x", "y", "z"]].values)
            processed_group.create_dataset("time", data=processed_df["time"].values)
            processed_group.create_dataset("labels", data=processed_df["label"].astype('S').values)
            processed_group.create_dataset("placement", data=processed_df["placement"].astype('S').values)

    print("✅ Raw and pre-processed data saved successfully to ./hdf5s/combined_dataset.h5")

