import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import random

import seaborn as sns
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import pickle

def BuildModel(iterations=1000, filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), "hdf5", "feature_dataset.h5"))):
    with h5py.File(filepath, 'r') as hdf:
        features = hdf['train/data'][:]
        labels = hdf['train/label'][:].astype(str)
    
    labelMaps = {'jumping': 1, 'walking': 0}
    
    y= np.array([labelMaps[label] for label in labels])
    
    featuresLabels =  [
        'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_variance', 'x_skew', 'x_kurtosis', 'x_rms', 'x_zcr',
        'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_variance', 'y_skew', 'y_kurtosis', 'y_rms', 'y_zcr',
        'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_variance', 'z_skew', 'z_kurtosis', 'z_rms', 'z_zcr'
    ]

    xTrain = pd.DataFrame(features, columns=featuresLabels)
    print(xTrain)

    model = LogisticRegression(max_iter=iterations)
    model.fit(xTrain,y)

    with open('Trained_Model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model

def TestModel(model, filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), "hdf5", "feature_dataset.h5"))):
    with h5py.File(filepath, 'r') as hdf:
        testFeatures = hdf['test/data'][:]
        testLabels = hdf['test/label'][:].astype(str)

    labelMaps = {'jumping': 1, 'walking': 0}
    y_test = np.array([labelMaps[label] for label in testLabels])

    featuresLabels =  [
        'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_variance', 'x_skew', 'x_kurtosis', 'x_rms', 'x_zcr',
        'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_variance', 'y_skew', 'y_kurtosis', 'y_rms', 'y_zcr',
        'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_variance', 'z_skew', 'z_kurtosis', 'z_rms', 'z_zcr'
    ]

    xTest = pd.DataFrame(testFeatures, columns=featuresLabels)


    pred = model.predict(xTest)
    prob = model.predict_proba(xTest)[:, 1]

    print("Predictions: ", pred)
    print("Confidence: ", prob)
    
    # accuracy
    accuracy = accuracy_score(y_test, pred)
    print("Accuracy: ", accuracy)
    
    #recall score
    recall = recall_score(y_test, pred)
    print("Recall:", recall)

    # CM 
    cm = confusion_matrix(y_test, pred)
    print("\nConfusion Matrix:")
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot()
    plt.show()

    #ROC stuff
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.show()

    # Calculating the AUC
    auc = roc_auc_score(y_test, prob)
    print('The AUC is:', auc)

    
    pass
def getModel():
    with open('Trained_Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def GenerateModel():
    BuildModel(iterations=100)
    model = getModel
    return model

def ModelStats():
    model = getModel()
    TestModel(model)

GenerateModel()
ModelStats()

