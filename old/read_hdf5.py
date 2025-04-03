import os
import h5py

# Automatically find the path to dataset.h5 in the current folder
file_path = os.path.join(os.path.dirname(__file__), "dataset.h5")

# Open the HDF5 file and read its contents
with h5py.File(file_path, "r") as hdf:
    print("✅ Opened dataset.h5\n")

    print("📂 Groups in file:")
    for group in hdf:
        print(f"  - {group}")

    print("\n📄 Datasets in 'raw':")
    for dataset in hdf["raw"]:
        data = hdf["raw"][dataset]
        print(f"  - {dataset}: shape = {data.shape}, dtype = {data.dtype}")

    # Example: read acceleration data and labels
    x = hdf["raw/data"][:]
    labels = hdf["raw/labels"][:].astype(str)

    print("\n🔎 First data row:", x[0])
    print("🔖 First label:", labels[0])