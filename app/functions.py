import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

def extractData(csv):
    df = pd.read_csv(csv)

    saveNum = 500  # 5 seconds at 100 Hz
    num_windows = len(df) // saveNum
    columns = ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']

    if not all(col in df.columns for col in columns):
        raise ValueError(f"CSV file must contain columns: {columns}")
    
    # Trim the DataFrame to a complete set of windows
    df = df.iloc[:num_windows * saveNum]

    # Extract only the relevant columns for processing
    data = df[columns].values
    windows = np.array(np.split(data, num_windows))
    return windows

def extract_features_from_window(window):
    df = pd.DataFrame(window, columns=['x', 'y', 'z'])

    def calculate_features(data):
        return pd.Series({
            'mean': np.mean(data),
            'max': np.max(data),
            'min': np.min(data),
            'range': np.max(data) - np.min(data),
            'variance': np.var(data),
            'std': np.std(data),
            'skew': skew(data),
            'kurtosis': kurtosis(data),
            'rms': np.sqrt(np.mean(data**2)),
            'zcr': ((np.diff(np.sign(data)) != 0).sum() / len(data))
        })

    x_features = calculate_features(df['x']).add_prefix('x_')
    y_features = calculate_features(df['y']).add_prefix('y_')
    z_features = calculate_features(df['z']).add_prefix('z_')

    features = pd.concat([x_features, y_features, z_features])
    return features

def predictAction(model, features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    return predictions
