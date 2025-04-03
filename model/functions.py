import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

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

def splitData():
    pass