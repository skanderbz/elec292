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

def makeHDF5(filepath='./hdf5/dataset.h5'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create a new HDF5 file (or overwrite if it already exists)
    with h5py.File(filepath, 'w') as hdf:
        # Create the three main groups
        hdf.create_group('raw')
        hdf.create_group('preprocessed')
        hdf.create_group('segmented')
        
    print(f"HDF5 file created at: {filepath}")

def loadrawData(filepath='./hdf5/dataset.h5', csv_dir='./csvdata/'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    users = ['skander', 'isaac', 'ahmed']
    activities = ['Jumping', 'Walking']
    
    with h5py.File(filepath, 'a') as hdf:
        raw_group = hdf.require_group('raw')

        for user in users:
            user_group = raw_group.require_group(user)

            # Lists to collect all data for the user
            data_list = []
            label_list = []
            placement_list = []
            time_list = []
            user_list = []

            for activity in activities:
                activity_folder = os.path.join(csv_dir, activity)

                if not os.path.exists(activity_folder):
                    print(f"❌ Folder not found: {activity_folder}")
                    continue

                for filename in os.listdir(activity_folder):
                    if user in filename.lower() and filename.endswith('.csv'):
                        
                        file_path = os.path.join(activity_folder, filename)
                        df = pd.read_csv(file_path, usecols=[0, 1, 2, 3])
                        df.columns = ['time', 'x', 'y', 'z']

                        position = extract_position(filename)
                        label = 'jumping' if activity == 'Jumping' else 'walking'

                        # Append data
                        data_list.append(df[['x', 'y', 'z']].values)
                        label_list.append([label.encode('utf-8')] * len(df))
                        placement_list.append([position.encode('utf-8')] * len(df))
                        time_list.append(df['time'].values)
                        user_list.append([user.encode('utf-8')] * len(df))

            # Combine all the collected data for the user
            if data_list:
                data_combined = np.vstack(data_list)
                labels_combined = np.concatenate(label_list)
                placement_combined = np.concatenate(placement_list)
                time_combined = np.concatenate(time_list)
                user_combined = np.concatenate(user_list)

                # Save all datasets for this user
                user_group.create_dataset('data', data=data_combined, compression='gzip')
                user_group.create_dataset('labels', data=labels_combined, compression='gzip')
                user_group.create_dataset('placement', data=placement_combined, compression='gzip')
                user_group.create_dataset('time', data=time_combined, compression='gzip')
                user_group.create_dataset('user', data=user_combined, compression='gzip')

                print(f"✅ Saved all data for {user}")

def extract_position(filename):
    filename = filename.lower()
    if 'backpocket' in filename:
        return 'backpocket'
    elif 'frontpocket' in filename:
        return 'frontpocket'
    elif 'hand' in filename:
        return 'hand'
    elif 'jacket' in filename:
        return 'jacket'
    else:
        return 'unknown'
    
def addPreProcessedData(filepath='./hdf5/dataset.h5', rolling_window_size=5):
    with h5py.File(filepath, 'a') as hdf:
        if 'preprocessed' not in hdf:
            hdf.create_group('preprocessed')
        
        raw_group = hdf['raw']
        preprocessed_group = hdf['preprocessed']
        
        for user in raw_group.keys():
            user_raw_group = raw_group[user]
            user_preprocessed_group = preprocessed_group.require_group(user)

            # Load datasets
            data = user_raw_group['data'][:]
            time = user_raw_group['time'][:]
            labels = user_raw_group['labels'][:].astype(str)
            placement = user_raw_group['placement'][:].astype(str)
            user_names = user_raw_group['user'][:].astype(str)
            
            # Convert data to DataFrame for preprocessing
            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            df['time'] = time
            df['label'] = labels
            df['placement'] = placement
            df['user'] = user_names

            # 1. Replace NaNs with Linear Interpolation
            df.interpolate(method='linear', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)

            # 2. Apply Rolling Average (Moving Average Filter)
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].rolling(window=rolling_window_size, min_periods=1).mean()
            
            # 3. Remove NaNs Created By Rolling Average (IMPORTANT STEP)
            df.dropna(subset=['x', 'y', 'z'], inplace=True)
            
            # Prepare arrays for saving
            data_array = df[['x', 'y', 'z']].values
            time_array = df['time'].values
            label_array = df['label'].astype('S').values
            placement_array = df['placement'].astype('S').values
            user_array = df['user'].astype('S').values
            
            # Save preprocessed data to the HDF5 file
            if 'data' in user_preprocessed_group:
                del user_preprocessed_group['data']
            if 'time' in user_preprocessed_group:
                del user_preprocessed_group['time']
            if 'labels' in user_preprocessed_group:
                del user_preprocessed_group['labels']
            if 'placement' in user_preprocessed_group:
                del user_preprocessed_group['placement']
            if 'user' in user_preprocessed_group:
                del user_preprocessed_group['user']
            
            user_preprocessed_group.create_dataset('data', data=data_array, compression='gzip')
            user_preprocessed_group.create_dataset('time', data=time_array, compression='gzip')
            user_preprocessed_group.create_dataset('labels', data=label_array, compression='gzip')
            user_preprocessed_group.create_dataset('placement', data=placement_array, compression='gzip')
            user_preprocessed_group.create_dataset('user', data=user_array, compression='gzip')
            
            print(f"✅ Preprocessed data saved for {user}")


def plot_raw_vs_preprocessed(user, activity, filepath='./hdf5/dataset.h5', max_time=None):
    with h5py.File(filepath, 'r') as hdf:
        # Load raw data
        raw_group = hdf[f'raw/{user}']
        
        raw_data = raw_group['data'][:]
        raw_time = raw_group['time'][:]
        raw_labels = raw_group['labels'][:].astype(str)
        
        # Filter the raw data to the desired activity (Jumping or Walking)
        raw_mask = (raw_labels == activity.lower())
        raw_df = pd.DataFrame(raw_data[raw_mask], columns=['x', 'y', 'z'])
        raw_df['time'] = raw_time[raw_mask]

        # Limit to the first max_time seconds if max_time is provided
        if max_time is not None:
            raw_df = raw_df[raw_df['time'] <= max_time]

        # Load preprocessed data
        preprocessed_group = hdf[f'preprocessed/{user}']
        
        preprocessed_data = preprocessed_group['data'][:]
        preprocessed_time = preprocessed_group['time'][:]
        preprocessed_labels = preprocessed_group['labels'][:].astype(str)
        
        # Filter the preprocessed data to the desired activity (Jumping or Walking)
        preprocessed_mask = (preprocessed_labels == activity.lower())
        preprocessed_df = pd.DataFrame(preprocessed_data[preprocessed_mask], columns=['x', 'y', 'z'])
        preprocessed_df['time'] = preprocessed_time[preprocessed_mask]

        # Limit to the first max_time seconds if max_time is provided
        if max_time is not None:
            preprocessed_df = preprocessed_df[preprocessed_df['time'] <= max_time]

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw data plot
        axes[0].plot(raw_df['time'], raw_df['x'], label='X-axis', alpha=0.6)
        axes[0].plot(raw_df['time'], raw_df['y'], label='Y-axis', alpha=0.6)
        axes[0].plot(raw_df['time'], raw_df['z'], label='Z-axis', alpha=0.6)
        axes[0].set_title(f'Raw Data for {user.title()} - {activity}')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Preprocessed data plot
        axes[1].plot(preprocessed_df['time'], preprocessed_df['x'], label='X-axis', alpha=0.6)
        axes[1].plot(preprocessed_df['time'], preprocessed_df['y'], label='Y-axis', alpha=0.6)
        axes[1].plot(preprocessed_df['time'], preprocessed_df['z'], label='Z-axis', alpha=0.6)
        axes[1].set_title(f'Preprocessed Data for {user.title()} - {activity}')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

#debug function
def diagnose_hdf5(filepath='./hdf5/dataset.h5', user='skander'):
    with h5py.File(filepath, 'r') as hdf:
        print(f"📂 HDF5 Keys: {list(hdf.keys())}")
        
        if f'raw/{user}' in hdf:
            print(f"\n🔑 Groups under raw/{user}: {list(hdf[f'raw/{user}'].keys())}")
            data = hdf[f'raw/{user}/data'][:]
            time = hdf[f'raw/{user}/time'][:]

            # Check for NaNs
            print(f"\n🔍 Checking NaNs in raw data for {user}")
            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            print(df.isna().sum())
            print(f"Number of rows: {len(df)}")
            
            # Check Data Types
            print("\n🔍 Data Types:")
            print(df.dtypes)
            
            # Check Data Range
            print("\n🔍 Data Range:")
            print(f"x: {df['x'].min()} to {df['x'].max()}")
            print(f"y: {df['y'].min()} to {df['y'].max()}")
            print(f"z: {df['z'].min()} to {df['z'].max()}")
            print(f"time: {time.min()} to {time.max()}")

#debug function
def debug_plot_skander_jumping():
    # Filepath to your HDF5 file
    filepath = './hdf5/dataset.h5'
    
    with h5py.File(filepath, 'r') as hdf:
        # Load data from HDF5 file
        data = hdf['raw/skander/data'][:]  # Note: No 'jumping' subgroup
        time = hdf['raw/skander/time'][:]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        df['time'] = time

        # Ensure the DataFrame index is reset properly
        df.reset_index(drop=True, inplace=True)

        # ✅ Sort DataFrame by time to ensure it's ordered properly
        df = df.sort_values(by='time').reset_index(drop=True)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['x'], label='X-axis', alpha=0.6)
    plt.plot(df['time'], df['y'], label='Y-axis', alpha=0.6)
    plt.plot(df['time'], df['z'], label='Z-axis', alpha=0.6)
    
    plt.title('Debug Plot - Raw Data (Skander)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    plt.show()

#debug function
def check_data_integrity(filepath='./hdf5/dataset.h5'):
    with h5py.File(filepath, 'r') as hdf:
        # Load data from HDF5 file
        data = hdf['raw/skander/data'][:]
        time = hdf['raw/skander/time'][:]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        df['time'] = time

    # Check for NaN values
    nan_check = df.isna().sum()
    print("🔍 NaN Check:")
    print(nan_check)
    
    # Check for Inf values
    inf_check = np.isinf(df).sum().sum()  # Sum over entire DataFrame
    print(f"\n🔍 Inf Check: {inf_check} Inf values detected.")

    # Get first and last value of time
    first_time = df['time'].iloc[0]
    last_time = df['time'].iloc[-1]
    
    print(f"\n🕒 First Time Value: {first_time}")
    print(f"🕒 Last Time Value: {last_time}")
    
    # Check if time is sorted
    is_sorted = df['time'].is_monotonic_increasing
    print(f"\n✅ Is Time Sorted? {is_sorted}")

def splitData(filepath='./hdf5/dataset.h5', window_size=500, train_ratio=0.9):
    with h5py.File(filepath, 'a') as hdf:  # Open in append mode to add segmented data
        # Ensure the segmented group exists
        segmented_group = hdf.require_group('segmented')
        
        # Create or open the train and test groups
        train_group = segmented_group.require_group('train')
        test_group = segmented_group.require_group('test')

        all_windows = []
        all_labels = []

        for user in hdf['preprocessed']:
            user_group = hdf[f'preprocessed/{user}']
            
            # Load the data
            data = user_group['data'][:]
            time = user_group['time'][:]
            labels = user_group['labels'][:].astype(str)
            positions = user_group['placement'][:].astype(str)
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            df['time'] = time
            df['label'] = labels
            df['placement'] = positions
            df['user'] = user  # Add user as a column to track origin
            
            # Generate windows
            for i in range(0, len(df) - window_size, window_size):
                window = df.iloc[i:i + window_size]
                label = window['label'].mode()[0]  # Use the most common label in the window
                user_label = window['user'].iloc[0]
                
                # Store both data and label
                all_windows.append(window[['x', 'y', 'z']].values)
                all_labels.append(f"{label}_{user_label}")  # Combine label and user (e.g., 'walking_skander')

        # Convert lists to numpy arrays
        all_windows = np.array(all_windows)  # Shape: (num_windows, window_size, 3)
        all_labels = np.array(all_labels).astype('S')  # Convert labels to bytes

        # Shuffle all data
        indices = np.arange(len(all_windows))
        np.random.shuffle(indices)
        
        all_windows = all_windows[indices]
        all_labels = all_labels[indices]

        # Split into training and testing datasets
        split_index = int(len(all_windows) * train_ratio)
        train_data, test_data = all_windows[:split_index], all_windows[split_index:]
        train_labels, test_labels = all_labels[:split_index], all_labels[split_index:]

        # 🔥 Remove existing datasets if they exist
        if 'data' in train_group:
            del train_group['data']
        if 'labels' in train_group:
            del train_group['labels']
        if 'data' in test_group:
            del test_group['data']
        if 'labels' in test_group:
            del test_group['labels']

        # ✅ Save the data
        train_group.create_dataset('data', data=train_data, compression='gzip')
        train_group.create_dataset('labels', data=train_labels, compression='gzip')

        test_group.create_dataset('data', data=test_data, compression='gzip')
        test_group.create_dataset('labels', data=test_labels, compression='gzip')

        print("✅ Data successfully split and saved to segmented group")


def save_segmented_to_csv(filepath='./hdf5/dataset.h5', output_dir='./csv_debug'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(filepath, 'r') as hdf:
        for split in ['train', 'test']:
            group = hdf[f'segmented/{split}']

            # Load data and labels
            data = group['data'][:]  # Shape: (num_windows, window_size, 3)
            labels = group['labels'][:].astype(str)  # Convert labels from bytes to strings

            # Flatten data for CSV writing
            flattened_data = data.reshape(data.shape[0], -1)  # Collapse window dimension
            
            # Create a DataFrame
            df = pd.DataFrame(flattened_data)
            df['label'] = labels  # Add labels as the last column
            
            # Save to CSV
            csv_file = os.path.join(output_dir, f'{split}_data.csv')
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Saved {split} data to {csv_file}")


def extract_features_from_window(window):
    """
    Extracts 10 features from a single window of data.
    Args:
        window (numpy array): 2D array of shape (window_size, 3), columns for x, y, z.
    Returns:
        pd.Series: A series containing the extracted features.
    """
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

def extract_and_save_features(hdf5_filepath, output_filepath='./hdf5/feature_dataset.h5', normalize_features=True):
    """
    Extracts features from raw data, normalizes them if specified, and saves to a new HDF5 file.
    
    Args:
        hdf5_filepath (str): Path to the original HDF5 file containing segmented data.
        output_filepath (str): Path to save the new HDF5 file containing features.
        normalize_features (bool): Whether to apply normalization before saving. Default is True.
    """
    with h5py.File(hdf5_filepath, 'r') as hdf, h5py.File(output_filepath, 'w') as new_hdf:
        for group_name in ['train', 'test']:
            data_group = hdf[f'segmented/{group_name}/data'][:]
            labels = hdf[f'segmented/{group_name}/labels'][:].astype(str)
            actions = [label.split('_')[0] for label in labels]
            
            features_list = []
            for window in data_group:
                features = extract_features_from_window(window)
                features_list.append(features.values)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # ✅ Normalize if specified
            if normalize_features:
                features_df = normalize(features_df)
            
            # Convert back to numpy array for saving
            feature_data = features_df.values
            labels_array = np.array(actions, dtype='S')
            action_array = np.array(labels, dtype='S')
            
            # Save features to new HDF5
            grp = new_hdf.create_group(group_name)
            grp.create_dataset('data', data=feature_data, compression='gzip')
            grp.create_dataset('label', data=labels_array, compression='gzip')
            grp.create_dataset('action', data=action_array, compression='gzip')
            
            print(f"✅ Features extracted, {'normalized, ' if normalize_features else ''}and saved for {group_name}")

    return features_df, labels

def normalize(df):
    """
    Normalize the features using Min-Max scaling to range [0, 1].
    
    Args:
        df (pd.DataFrame): The DataFrame containing extracted features.
        
    Returns:
        pd.DataFrame: A normalized DataFrame.
    """
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

def visualize_features(filepath='./hdf5/feature_dataset.h5', group_name='train', feature_subset=None):
    """
    Visualizes the extracted features for jumping and walking in a clear way, divided into separate plots for X, Y, and Z.
    
    Args:
        filepath (str): Path to the HDF5 file containing features.
        group_name (str): 'train' or 'test' - which group to visualize.
        feature_subset (list): A list of specific features to visualize. Default is all features.
    """
    # Load the features and labels from the HDF5 file
    with h5py.File(filepath, 'r') as hdf:
        features = hdf[f'{group_name}/data'][:]
        labels = hdf[f'{group_name}/label'][:].astype(str)

    # Convert the features to a DataFrame
    feature_names = [
        'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_variance', 'x_skew', 'x_kurtosis', 'x_rms', 'x_zcr',
        'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_variance', 'y_skew', 'y_kurtosis', 'y_rms', 'y_zcr',
        'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_variance', 'z_skew', 'z_kurtosis', 'z_rms', 'z_zcr'
    ]
    
    df = pd.DataFrame(features, columns=feature_names)
    df['label'] = labels  # Add the labels to the DataFrame

    # Separate features by axis
    x_features = [feat for feat in feature_names if feat.startswith('x_')]
    y_features = [feat for feat in feature_names if feat.startswith('y_')]
    z_features = [feat for feat in feature_names if feat.startswith('z_')]

    # Filter to only the desired features if specified
    if feature_subset:
        x_features = [feat for feat in x_features if feat in feature_subset]
        y_features = [feat for feat in y_features if feat in feature_subset]
        z_features = [feat for feat in z_features if feat in feature_subset]

    # Function to plot a group of features
    def plot_features(features, title):
        plt.figure(figsize=(20, 15))
        sns.set(style="whitegrid")
        num_features = len(features)
        cols = 3  # Number of columns in the plot grid
        rows = (num_features + cols - 1) // cols  # Calculate the number of rows required

        for idx, feature in enumerate(features):
            plt.subplot(rows, cols, idx + 1)
            sns.boxplot(x='label', y=feature, data=df)
            #plt.title(f'{feature} by Label')
            plt.xlabel('Label')
            plt.ylabel(feature)
        
        plt.suptitle(title, fontsize=20)
        plt.tight_layout()
        plt.show()
    
    # Plot the X, Y, and Z features separately
    plot_features(x_features, "X-Axis Features")
    plot_features(y_features, "Y-Axis Features")
    plot_features(z_features, "Z-Axis Features")

def end():
    pass