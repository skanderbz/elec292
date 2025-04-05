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
    
    with h5py.File(filepath, 'w') as hdf:
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

            data_list = []
            label_list = []
            placement_list = []
            time_list = []
            user_list = []

            for activity in activities:
                activity_folder = os.path.join(csv_dir, activity)

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

            data = user_raw_group['data'][:]
            time = user_raw_group['time'][:]
            labels = user_raw_group['labels'][:].astype(str)
            placement = user_raw_group['placement'][:].astype(str)
            user_names = user_raw_group['user'][:].astype(str)

            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            df['time'] = time
            df['label'] = labels
            df['placement'] = placement
            df['user'] = user_names

            df.interpolate(method='linear', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].rolling(window=rolling_window_size, min_periods=1).mean()
            
            #Remove NaNs
            df.dropna(subset=['x', 'y', 'z'], inplace=True)

            data_array = df[['x', 'y', 'z']].values
            time_array = df['time'].values
            label_array = df['label'].astype('S').values
            placement_array = df['placement'].astype('S').values
            user_array = df['user'].astype('S').values
            
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


def plot_raw_vs_preprocessed(user, activity, filepath='./hdf5/dataset.h5', max_time=None):
    with h5py.File(filepath, 'r') as hdf:
        # Load raw data
        raw_group = hdf[f'raw/{user}']
        
        raw_data = raw_group['data'][:]
        raw_time = raw_group['time'][:]
        raw_labels = raw_group['labels'][:].astype(str)
        
        raw_mask = (raw_labels == activity.lower())
        raw_df = pd.DataFrame(raw_data[raw_mask], columns=['x', 'y', 'z'])
        raw_df['time'] = raw_time[raw_mask]

        if max_time is not None:
            raw_df = raw_df[raw_df['time'] <= max_time]

        preprocessed_group = hdf[f'preprocessed/{user}']
        
        preprocessed_data = preprocessed_group['data'][:]
        preprocessed_time = preprocessed_group['time'][:]
        preprocessed_labels = preprocessed_group['labels'][:].astype(str)

        preprocessed_mask = (preprocessed_labels == activity.lower())
        preprocessed_df = pd.DataFrame(preprocessed_data[preprocessed_mask], columns=['x', 'y', 'z'])
        preprocessed_df['time'] = preprocessed_time[preprocessed_mask]

        if max_time is not None:
            preprocessed_df = preprocessed_df[preprocessed_df['time'] <= max_time]

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
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


def splitData(filepath='./hdf5/dataset.h5', window_size=500, train_ratio=0.9):
    with h5py.File(filepath, 'a') as hdf:  
        segmented_group = hdf.require_group('segmented')
        
        train_group = segmented_group.require_group('train')
        test_group = segmented_group.require_group('test')

        all_windows = []
        all_labels = []

        for user in hdf['preprocessed']:
            user_group = hdf[f'preprocessed/{user}']
            
            data = user_group['data'][:]
            time = user_group['time'][:]
            labels = user_group['labels'][:].astype(str)
            positions = user_group['placement'][:].astype(str)

            df = pd.DataFrame(data, columns=['x', 'y', 'z'])
            df['time'] = time
            df['label'] = labels
            df['placement'] = positions
            df['user'] = user 

            for i in range(0, len(df) - window_size, window_size):
                window = df.iloc[i:i + window_size]
                label = window['label'].mode()[0] 
                user_label = window['user'].iloc[0]

                all_windows.append(window[['x', 'y', 'z']].values)
                all_labels.append(f"{label}_{user_label}") 

        all_windows = np.array(all_windows)  
        all_labels = np.array(all_labels).astype('S')  

        # Shuffle all data
        indices = np.arange(len(all_windows))
        np.random.shuffle(indices)
        
        all_windows = all_windows[indices]
        all_labels = all_labels[indices]
        split_index = int(len(all_windows) * train_ratio)
        train_data, test_data = all_windows[:split_index], all_windows[split_index:]
        train_labels, test_labels = all_labels[:split_index], all_labels[split_index:]

        if 'data' in train_group:
            del train_group['data']
        if 'labels' in train_group:
            del train_group['labels']
        if 'data' in test_group:
            del test_group['data']
        if 'labels' in test_group:
            del test_group['labels']

        train_group.create_dataset('data', data=train_data, compression='gzip')
        train_group.create_dataset('labels', data=train_labels, compression='gzip')

        test_group.create_dataset('data', data=test_data, compression='gzip')
        test_group.create_dataset('labels', data=test_labels, compression='gzip')

# FOR CHECKING FOR OVERFILLING
def save_segmented_to_csv(filepath='./hdf5/dataset.h5', output_dir='./csv_debug'):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(filepath, 'r') as hdf:
        for split in ['train', 'test']:
            group = hdf[f'segmented/{split}']

            data = group['data'][:] 
            labels = group['labels'][:].astype(str)  
            flattened_data = data.reshape(data.shape[0], -1)  

            df = pd.DataFrame(flattened_data)
            df['label'] = labels 
            
            csv_file = os.path.join(output_dir, f'{split}_data.csv')
            df.to_csv(csv_file, index=False)


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

def extract_and_save_features(hdf5_filepath, output_filepath='./hdf5/feature_dataset.h5', normalize_features=True):

    with h5py.File(hdf5_filepath, 'r') as hdf, h5py.File(output_filepath, 'w') as new_hdf:
        for group_name in ['train', 'test']:
            data_group = hdf[f'segmented/{group_name}/data'][:]
            labels = hdf[f'segmented/{group_name}/labels'][:].astype(str)
            actions = [label.split('_')[0] for label in labels]
            
            features_list = []
            for window in data_group:
                features = extract_features_from_window(window)
                features_list.append(features.values)

            features_df = pd.DataFrame(features_list)
            
            if normalize_features:
                features_df = normalize(features_df)
            
            feature_data = features_df.values
            labels_array = np.array(actions, dtype='S')
            action_array = np.array(labels, dtype='S')
            
            grp = new_hdf.create_group(group_name)
            grp.create_dataset('data', data=feature_data, compression='gzip')
            grp.create_dataset('label', data=labels_array, compression='gzip')
            grp.create_dataset('action', data=action_array, compression='gzip')

    return features_df, labels

def normalize(df):
    scaler = StandardScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    return df_norm

def visualize_features(filepath='./hdf5/feature_dataset.h5', group_name='train', feature_subset=None):
    with h5py.File(filepath, 'r') as hdf:
        features = hdf[f'{group_name}/data'][:]
        labels = hdf[f'{group_name}/label'][:].astype(str)

    feature_names = [
        'x_mean', 'x_std', 'x_max', 'x_min', 'x_range', 'x_variance', 'x_skew', 'x_kurtosis', 'x_rms', 'x_zcr',
        'y_mean', 'y_std', 'y_max', 'y_min', 'y_range', 'y_variance', 'y_skew', 'y_kurtosis', 'y_rms', 'y_zcr',
        'z_mean', 'z_std', 'z_max', 'z_min', 'z_range', 'z_variance', 'z_skew', 'z_kurtosis', 'z_rms', 'z_zcr'
    ]
    
    df = pd.DataFrame(features, columns=feature_names)
    df['label'] = labels 
    x_features = [feat for feat in feature_names if feat.startswith('x_')]
    y_features = [feat for feat in feature_names if feat.startswith('y_')]
    z_features = [feat for feat in feature_names if feat.startswith('z_')]
    if feature_subset:
        x_features = [feat for feat in x_features if feat in feature_subset]
        y_features = [feat for feat in y_features if feat in feature_subset]
        z_features = [feat for feat in z_features if feat in feature_subset]

    def plot_features(features, title):
        plt.figure(figsize=(20, 15))
        sns.set(style="whitegrid")
        num_features = len(features)
        cols = 3 
        rows = (num_features + cols - 1) // cols 

        for idx, feature in enumerate(features):
            plt.subplot(rows, cols, idx + 1)
            sns.boxplot(x='label', y=feature, data=df)
            #plt.title(f'{feature} by Label')
            plt.xlabel('Label')
            plt.ylabel(feature)
        
        plt.suptitle(title, fontsize=20)
        plt.tight_layout()
        plt.show()
    plot_features(x_features, "X-Axis Features")
    plot_features(y_features, "Y-Axis Features")
    plot_features(z_features, "Z-Axis Features")

def end():
    #hawk tuah 
    pass