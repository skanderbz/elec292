import functions as f
import h5py
import os
import numpy as np
import pandas as pd

#f.makeHDF5()
#f.loadrawData()

#f.addPreProcessedData()
#f.diagnose_hdf5(user='skander')

def check_hdf5_structure(filepath='./hdf5/dataset.h5'):
    with h5py.File(filepath, 'r') as hdf:
        def print_structure(name, obj):
            print(name)
        hdf.visititems(print_structure)
#check_hdf5_structure()     
#f.plot_raw_vs_preprocessed('skander','Jumping',max_time=10)
#f.debug_plot_skander_jumping()
#f.check_data_integrity('./hdf5/dataset.h5')
#f.addPreProcessedData
#f.splitData()
#f.save_segmented_to_csv()
#f.extract_and_save_features(hdf5_filepath='./hdf5/dataset.h5',normalize_features=True)

f.visualize_features()