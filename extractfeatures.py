import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os

def df_from_processedData(processedData=None):
    if not processedData:
        processedData = os.path.join("hdf5s", "processed_dataset.h5")
    
    with h5py.File(processedData, "r") as hdf:
        data = hdf['processed/data'][:]
        time = hdf['processed/time'][:]
        labels = hdf['processed/labels'][:].astype(str)
        user = hdf['processed/user'][:].astype(str)
        placement = hdf['processed/placement'][:].astype(str)

    df = pd.DataFrame({
        "time": time,
        "x":data[:,0],
        "y":data[:,1],
        "z":data[:,2],
        "label": labels,
        "user": user,
        "placement": placement
    })

    return df

def Extract_from_df(df=None):
    if not df:
        df = df_from_processedData()
    
    walkingdf = df[df['label'] == 'walking']
    jumpingdf = df[df['label'] == 'jumping']

    return walkingdf , jumpingdf

def extractFeatures(df,windowsize=5):
    features = pd.DataFrame(columns=['mean','std','max','skew','kurtosis','min'])
    
    features['mean'] = 0
    pass




