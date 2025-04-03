import h5py

def visualize_hdf5(file_path):
    """Recursively prints the structure of an HDF5 file."""
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"📂 Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"📄 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

    with h5py.File(file_path, 'r') as hdf:
        print(f"\n📁 Visualizing {file_path}\n")
        hdf.visititems(print_structure)
        print("\n✅ Visualization Complete!\n")

# Visualize your combined dataset
visualize_hdf5('./hdf5s/combined_dataset.h5')