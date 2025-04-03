import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load the HDF5 files
raw_file_path = os.path.join(os.path.dirname(__file__),"hdf5s", "dataset.h5")
processed_file_path = os.path.join(os.path.dirname(__file__),"hdf5s", "processed_dataset.h5")

def load_data(file_path, group_name):
    with h5py.File(file_path, "r") as hdf:
        # Load datasets
        data = hdf[f"{group_name}/data"][:]          # shape: (N, 3)
        time = hdf[f"{group_name}/time"][:]          # shape: (N,)
        labels = hdf[f"{group_name}/labels"][:].astype(str)
        users = hdf[f"{group_name}/user"][:].astype(str)
        placement = hdf[f"{group_name}/placement"][:].astype(str)

    # Convert to NumPy structured array for filtering
    dataset = np.rec.fromarrays(
        [time, data[:, 0], data[:, 1], data[:, 2], labels, users, placement],
        names=["time", "x", "y", "z", "label", "user", "placement"]
    )

    return dataset

# Load both datasets
raw_data = load_data(raw_file_path, "raw")
processed_data = load_data(processed_file_path, "processed")

# 🔍 Filter by activity and user
activity = "jumping"
person = "skander"

raw_filtered = raw_data[(raw_data.label == activity) & (raw_data.user == person)]
processed_filtered = processed_data[(processed_data.label == activity) & (processed_data.user == person)]

"""# Limit to the first 1000 rows (optional)
raw_filtered = raw_filtered[:1000]
processed_filtered = processed_filtered[:1000]"""

#All rows
raw_filtered = raw_filtered
processed_filtered = processed_filtered


# 📈 Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Raw Data
ax1.plot(raw_filtered.time, raw_filtered.x, label="X-axis", alpha=0.6)
ax1.plot(raw_filtered.time, raw_filtered.y, label="Y-axis", alpha=0.6)
ax1.plot(raw_filtered.time, raw_filtered.z, label="Z-axis", alpha=0.6)
ax1.set_title(f"Raw Data for {person.title()} - {activity.title()}")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Acceleration (m/s²)")
ax1.legend()
ax1.grid(True)

# Plot Processed Data
ax2.plot(processed_filtered.time, processed_filtered.x, label="X-axis", alpha=0.6)
ax2.plot(processed_filtered.time, processed_filtered.y, label="Y-axis", alpha=0.6)
ax2.plot(processed_filtered.time, processed_filtered.z, label="Z-axis", alpha=0.6)
ax2.set_title(f"Processed Data for {person.title()} - {activity.title()}")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Acceleration (m/s²)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
