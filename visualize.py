import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load the HDF5 file
file_path = os.path.join(os.path.dirname(__file__), "dataset.h5")

with h5py.File(file_path, "r") as hdf:
    # Load datasets
    data = hdf["raw/data"][:]           # shape: (N, 3)
    time = hdf["raw/time"][:]           # shape: (N,)
    labels = hdf["raw/labels"][:].astype(str)
    users = hdf["raw/user"][:].astype(str)
    placement = hdf["raw/placement"][:].astype(str)

# Convert to NumPy structured array for filtering
dataset = np.rec.fromarrays(
    [time, data[:, 0], data[:, 1], data[:, 2], labels, users, placement],
    names=["time", "x", "y", "z", "label", "user", "placement"]
)

# 🔍 Filter by activity and user
activity = "jumping"
person = "skander"
filtered = dataset[(dataset.label == activity) & (dataset.user == person)]

# Limit to the first 1000 rows (optional)
filtered = filtered[:1000]

# 📈 Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered.time, filtered.x, label="X-axis")
plt.plot(filtered.time, filtered.y, label="Y-axis")
plt.plot(filtered.time, filtered.z, label="Z-axis")

plt.title(f"Acceleration for {person.title()} - {activity.title()}")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()