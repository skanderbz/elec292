import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import pickle

def csvtodf(csv):
    """
    Makes a DataFrame from a CSV file containing only time, x, y, and z data.
    
    Args:
        csv (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame with columns ['time', 'x', 'y', 'z'].
    """
    df = pd.read_csv(csv, usecols=[0, 1, 2, 3])
    df.columns = ['time', 'x', 'y', 'z']
    return df

def preprocessdf(df, rolling_window_size=5):
    """
    Preprocesses the input DataFrame exactly as done in addPreProcessedData.
    
    Steps:
      1. Replace NaNs with linear interpolation.
      2. Fill any remaining NaNs using backward and forward fill.
      3. Apply a rolling average filter (moving average) on the 'x', 'y', 'z' columns.
      4. Remove any NaNs created by the rolling average.
      
    Args:
        df (pd.DataFrame): Input DataFrame with columns ['time', 'x', 'y', 'z', 'label', 'placement', 'user'].
        rolling_window_size (int): Window size for the rolling average filter.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # 1. Replace NaNs with linear interpolation.
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # 2. Apply a rolling average filter to the accelerometer data.
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].rolling(window=rolling_window_size, min_periods=1).mean()
    
    # 3. Remove any NaNs that might have been created by the rolling average.
    df.dropna(subset=['x', 'y', 'z'], inplace=True)
    
    return df

def preprocess_df(df, rolling_window_size=500):
    """
    Preprocesses the input DataFrame by:
      1. Replacing NaNs using linear interpolation and filling remaining NaNs with backfill and forward fill.
      2. Applying a rolling average (moving average filter) on the 'x', 'y', and 'z' columns.
      3. Dropping any rows that still have NaNs in the accelerometer data.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['time', 'x', 'y', 'z'].
        rolling_window_size (int): Window size for the rolling average filter.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # 1. Replace NaNs with linear interpolation and fill any remaining gaps.
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # 2. Apply a rolling average filter to smooth the 'x', 'y', and 'z' data.
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].rolling(window=rolling_window_size, min_periods=1).mean()
    
    # 3. Remove any rows that still contain NaNs in the accelerometer data.
    df.dropna(subset=['x', 'y', 'z'], inplace=True)
    
    return df

def split_and_extract_features(df, window_size=500, normalize_features=True):
    """
    Splits the input DataFrame into non-overlapping windows and extracts features from each window.
    Optionally normalizes the resulting features using min–max scaling.

    Args:
        df (pd.DataFrame): DataFrame with columns ['time', 'x', 'y', 'z'].
        window_size (int): Number of rows per window (default 500, representing 5 seconds at 100Hz).
        normalize_features (bool): Whether to normalize the extracted features (default True).

    Returns:
        pd.DataFrame: A DataFrame where each row contains the extracted (and normalized) features for one window.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import skew, kurtosis

    def extract_features_from_window(window):
        """
        Extracts 10 features from a single window of data.
        For each axis (x, y, z), it computes:
            mean, std, max, min, range, variance, skew, kurtosis, rms, and zcr.
        """
        df_window = pd.DataFrame(window, columns=['x', 'y', 'z'])
        
        def calculate_features(data):
            return pd.Series({
                'mean': np.mean(data),
                'std': np.std(data),
                'max': np.max(data),
                'min': np.min(data),
                'range': np.max(data) - np.min(data),
                'variance': np.var(data),
                'skew': skew(data),
                'kurtosis': kurtosis(data),
                'rms': np.sqrt(np.mean(data**2)),
                'zcr': ((np.diff(np.sign(data)) != 0).sum() / len(data))
            })
        
        x_features = calculate_features(df_window['x']).add_prefix('x_')
        y_features = calculate_features(df_window['y']).add_prefix('y_')
        z_features = calculate_features(df_window['z']).add_prefix('z_')
        features = pd.concat([x_features, y_features, z_features])
        return features
    
    features_list = []
    n_rows = len(df)
    # Iterate in steps of window_size, discarding any incomplete window.
    for i in range(0, n_rows - window_size + 1, window_size):
        window = df.iloc[i:i + window_size]
        features = extract_features_from_window(window[['x', 'y', 'z']].values)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(features_df)
    
    # Normalize param
    if normalize_features:
        scaler = StandardScaler()
        features_df = pd.DataFrame(
            scaler.fit_transform(features_df),
            columns=features_df.columns,
            index=features_df.index
        )
    
    return features_df

def predict_activity(features_df, model_path="../model/Trained_Model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Ensure the features are in the correct order
    featuresLabels = [
        'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_variance', 'x_skew', 'x_kurtosis', 'x_rms', 'x_zcr',
        'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_variance', 'y_skew', 'y_kurtosis', 'y_rms', 'y_zcr',
        'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_variance', 'z_skew', 'z_kurtosis', 'z_rms', 'z_zcr'
    ]
    features_df = features_df[featuresLabels]
    
    # Use the model to predict activity for each window
    predictions = model.predict(features_df)
    #print("Predictions:", predictions)
    return predictions
