import h5py
import numpy as np
import pandas as pd

# Parameters for preprocessing
WINDOW_SIZE = 5  # Moving average window size

def apply_moving_average(data, window_size):
    """Apply a moving average filter to each axis of the data."""
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    smoothed = df.rolling(window=window_size, min_periods=1).mean()  # Apply moving average
    return smoothed.values  # Return as numpy array

def detect_and_handle_nans(data):
    """Detect NaN values and handle them with interpolation."""
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    
    # Detect NaNs
    if df.isna().sum().sum() > 0:
        print(f"⚠️ Detected NaN values. Replacing with interpolated values...")

        # Replace NaN values by linear interpolation
        df.interpolate(method='linear', inplace=True)
        
        # Fill any remaining NaNs (e.g., start or end of data) with the closest valid value
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
    else:
        print("✅ No NaN values detected.")
    
    return df.values

def preprocess_hdf5(input_file, output_file):
    with h5py.File(input_file, 'r') as hdf_in, h5py.File(output_file, 'w') as hdf_out:
        
        # Load raw data
        raw_data = hdf_in['raw/data'][:]
        time = hdf_in['raw/time'][:]
        labels = hdf_in['raw/labels'][:]
        users = hdf_in['raw/user'][:]
        placements = hdf_in['raw/placement'][:]

        # Convert labels, users, placements to strings
        labels = [label.decode() if isinstance(label, bytes) else label for label in labels]
        users = [user.decode() if isinstance(user, bytes) else user for user in users]
        placements = [placement.decode() if isinstance(placement, bytes) else placement for placement in placements]

        # Detect and handle NaNs
        print("🔍 Checking for NaN values...")
        raw_data = detect_and_handle_nans(raw_data)
        
        # Apply moving average filter to data
        print("📉 Applying moving average filter...")
        processed_data = apply_moving_average(raw_data, WINDOW_SIZE)

        # Save processed data to a new HDF5 file
        hdf_out.create_dataset("processed/data", data=processed_data)
        hdf_out.create_dataset("processed/time", data=time)
        hdf_out.create_dataset("processed/labels", data=np.array(labels, dtype='S'))
        hdf_out.create_dataset("processed/user", data=np.array(users, dtype='S'))
        hdf_out.create_dataset("processed/placement", data=np.array(placements, dtype='S'))

        print(f"✅ Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_hdf5("dataset.h5", "processed_dataset.h5")
