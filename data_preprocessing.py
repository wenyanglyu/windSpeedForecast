import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle


def load_csv(file_id):
    """
    Downloads the CSV from Google Drive and loads it into a DataFrame.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    with open("wind_data.csv", "wb") as file:
        file.write(response.content)

    # Load data
    wind_data = pd.read_csv("wind_data.csv", dtype=str)  # Read as string to detect non-numeric values
    return wind_data


def normalize_feature(series):
    """
    Normalizes a given series using its min and max values.
    """
    min_val = series.min(skipna=True)
    max_val = series.max(skipna=True)
    return (series - min_val) / (max_val - min_val)


CUT_IN_SPEED = 3.0  # m/s
RATED_SPEED = 15.0  # m/s
CUT_OUT_SPEED = 20.0  # m/s

'''
def normalize_wind_speed(wind_speed_series, min_speed=None, max_speed=None):
    """
    Custom normalization for wind speed based on the dynamic min and max values from the data.
    If wind speed is NaN, it returns NaN.
    """
    # Dynamically calculate the min and max of the wind speed series if not provided
    if min_speed is None or max_speed is None:
        min_speed = wind_speed_series.min()
        max_speed = wind_speed_series.max()

    # Use np.where to vectorize the conditional normalization
    normalized_speed = np.where(
        wind_speed_series.isna(),  # If the value is NaN
        np.nan,  # Return NaN for NaN wind speeds
        np.where(
            wind_speed_series < CUT_IN_SPEED,
            (wind_speed_series / CUT_IN_SPEED) * 0.1,
            np.where(
                (wind_speed_series >= CUT_IN_SPEED) & (wind_speed_series <= RATED_SPEED),
                0.1 + ((wind_speed_series - CUT_IN_SPEED) / (RATED_SPEED - CUT_IN_SPEED)) * 0.8,
                np.where(
                    (wind_speed_series > RATED_SPEED) & (wind_speed_series <= CUT_OUT_SPEED),
                    0.9 + ((wind_speed_series - RATED_SPEED) / (CUT_OUT_SPEED - RATED_SPEED)) * 0.1,
                    1.0 + ((wind_speed_series - CUT_OUT_SPEED) / (max_speed - CUT_OUT_SPEED)) * 0.1
                )
            )
        )
    )

    return normalized_speed
'''

def normalize_wind_speed(wind_speed_series):
    """
    Min-Max normalization for wind speed, handling NaN values correctly.
    Normalized range: [0,1].
    """
    # Compute min and max, ignoring NaN values
    min_speed = wind_speed_series.min(skipna=True)
    max_speed = wind_speed_series.max(skipna=True)

    # Apply Min-Max normalization while preserving NaNs
    return (wind_speed_series - min_speed) / (max_speed - min_speed)



def expand_greymouth_data(greymouth_data):
    """
    Expands the hourly data from Greymouth into 10-minute intervals.
    Each 1-hour data point is repeated 6 times for each 10-minute interval.
    """
    # Convert OBS_DATE to datetime
    greymouth_data['OBS_DATE'] = pd.to_datetime(greymouth_data['OBS_DATE'], errors='coerce')

    # List to hold expanded rows
    expanded_data = []

    # Loop through each row in the Greymouth data
    for _, row in greymouth_data.iterrows():
        # Get the timestamp and pressure
        timestamp = row['OBS_DATE']
        pressure = row['PRESSURE_MSL60']

        # Create 6 entries for each 10-minute interval in the hour
        for i in range(6):  # 6 intervals in 1 hour (e.g., 20:00, 20:10, ..., 20:50)
            interval_timestamp = timestamp + pd.Timedelta(minutes=10 * i)
            expanded_data.append([interval_timestamp, pressure])

    # Create a new DataFrame from the expanded data
    expanded_greymouth = pd.DataFrame(expanded_data, columns=['OBS_DATE', 'PRESSURE_MSL60'])

    # Ensure correct ordering by OBS_DATE
    expanded_greymouth = expanded_greymouth.sort_values(by='OBS_DATE').reset_index(drop=True)

    return expanded_greymouth


def preprocess_data(nelson_data, greymouth_data):
    """
    Preprocesses the data by expanding Greymouth data to 10-minute intervals,
    merging with Nelson data, and performing normalization.
    """
    # Ensure that both datasets have 'OBS_DATE' as datetime format
    nelson_data['OBS_DATE'] = pd.to_datetime(nelson_data['OBS_DATE'], errors='coerce')
    greymouth_data['OBS_DATE'] = pd.to_datetime(greymouth_data['OBS_DATE'], errors='coerce')

    # Remove timezone information (or standardize to UTC)
    nelson_data['OBS_DATE'] = nelson_data['OBS_DATE'].dt.tz_localize(None)
    greymouth_data['OBS_DATE'] = greymouth_data['OBS_DATE'].dt.tz_localize(None)

    # Expand Greymouth data to 10-minute intervals
    expanded_greymouth = expand_greymouth_data(greymouth_data)

    # Merge the two datasets on OBS_DATE
    combined_data = pd.merge(nelson_data, expanded_greymouth[['OBS_DATE', 'PRESSURE_MSL60']], on='OBS_DATE', how='left')

    # Convert relevant columns to numeric values (excluding MEAN_SPD10 for now)
    numerical_cols = ['PRESSURE_MSL60', 'PRESSURE_MSL10', 'RAD_GLOBAL10', 'RAINFALL10', 'MEAN_TEMP10', 'MEAN_RELHUM10']
    combined_data[numerical_cols] = combined_data[numerical_cols].apply(pd.to_numeric, errors='coerce')

    # Apply forward fill to all numerical columns except MEAN_SPD10
    combined_data[numerical_cols] = combined_data[numerical_cols].ffill()

    # Convert MEAN_SPD10 to numeric and set invalid values to NaN
    combined_data['MEAN_SPD10'] = pd.to_numeric(combined_data['MEAN_SPD10'], errors='coerce')

    # Mark MEAN_SPD10 as invalid if NaN (no forward filling for wind speed)
    combined_data['valid_for_window'] = ~combined_data['MEAN_SPD10'].isna()

    # Normalize MEAN_SPD10 using the custom wind speed normalization function
    combined_data['normalized_MEAN_SPD10'] = normalize_wind_speed(combined_data['MEAN_SPD10'])

    # Normalize PRESSURE_MSL60 and other columns using Min-Max scaling
    for col in numerical_cols:
        combined_data[f'normalized_{col}'] = normalize_feature(combined_data[col])

    import numpy as np

    # Convert MEAN_DIR10 to numeric, forcing errors to NaN
    combined_data['MEAN_DIR10'] = pd.to_numeric(combined_data['MEAN_DIR10'], errors='coerce')

    # Convert MEAN_DIR10 to radians
    combined_data['wind_dir_rad'] = np.radians(combined_data['MEAN_DIR10'])

    # Apply sin & cos transformation
    combined_data['wind_dir_sin'] = np.sin(combined_data['wind_dir_rad'])
    combined_data['wind_dir_cos'] = np.cos(combined_data['wind_dir_rad'])

    # Combine sin and cos using atan2 (arctangent function)
    combined_data['wind_dir_circular'] = np.arctan2(combined_data['wind_dir_sin'], combined_data['wind_dir_cos'])

    # Normalize the circular wind direction to [0,1]
    combined_data['wind_dir_circular_norm'] = (combined_data['wind_dir_circular'] + np.pi) / (2 * np.pi)

    combined_data = combined_data.drop(columns=['wind_dir_sin', 'wind_dir_cos', 'wind_dir_rad', 'wind_dir_circular'])

    # Extract time features
    combined_data['year'] = combined_data['OBS_DATE'].dt.year
    combined_data['day_of_year'] = combined_data['OBS_DATE'].dt.dayofyear
    combined_data['minute_of_day'] = combined_data['OBS_DATE'].dt.hour * 60 + combined_data['OBS_DATE'].dt.minute

    # Normalize year (2015-2025 -> 0-1)
    combined_data['normalized_year'] = (combined_data['year'] - 2015) / (2025 - 2015)

    # Normalize day of year and time within a day using sine function
    combined_data['day_of_year_sin'] = np.sin(2 * np.pi * (combined_data['day_of_year'] - 245) / 365)  # 245 is Sept 1st
    combined_data['minute_of_day_sin'] = np.sin(2 * np.pi * (combined_data['minute_of_day'] - 540) / 1440)  # 540 is 9:00 AM

    # Convert MEAN_SPD10 to numeric and set invalid values to NaN
    combined_data['MEAN_SPD10'] = pd.to_numeric(combined_data['MEAN_SPD10'], errors='coerce')

    # Mark invalid wind speed data as False in the valid_for_window column
    combined_data['valid_for_window'] = ~combined_data['MEAN_SPD10'].isna()

    # Selecting processed columns
    selected_features = ['normalized_year', 'day_of_year_sin', 'minute_of_day_sin'] + \
                        [f'normalized_{col}' for col in numerical_cols] + \
                        ['wind_dir_circular_norm', 'normalized_MEAN_SPD10', 'valid_for_window']

    return combined_data[selected_features]

def create_sliding_windows(wind_data, window_size=2016, forecast_size=144, shift=6, features=None):
    """
    Creates sliding windows for time series forecasting while avoiding null-speed days.
    Uses a specified shift value to control step size when moving the window.

    - First 0-7 days: Min, Max, and Average for each day for each feature.
    - Next 8-11 days: Hourly averages for each feature.
    - Last 3 days: 10-minute interval values for each feature.
    """
    dataset_values = wind_data.drop(columns=['valid_for_window']).values
    valid_mask = wind_data['valid_for_window'].values  # Mask for valid windows

    X, y = [], []

    # Loop over the dataset to create sliding windows
    for i in range(0, len(dataset_values) - window_size - forecast_size, shift):
        # Check if the entire window is valid (no invalid data)
        if not np.any(valid_mask[i:i + window_size + forecast_size] == False):

            # Extract the features for the sliding window
            window_data = []

            for day in range(7):
                day_start = i + day * 144  # 每天144个时间间隔
                day_end = day_start + 144
                day_data = dataset_values[day_start:day_end, :]  # 每天144个数据点

                # 计算min、max、mean
                min_vals = np.min(day_data, axis=0)
                max_vals = np.max(day_data, axis=0)
                mean_vals = np.mean(day_data, axis=0)

                # Ensure the length is 11 (features) and append each set of 11 values separately
                if len(min_vals) == 11 and len(max_vals) == 11 and len(mean_vals) == 11:
                    window_data.extend(min_vals)  # Add min values (11 values)
                    window_data.extend(max_vals)  # Add max values (11 values)
                    window_data.extend(mean_vals)  # Add mean values (11 values)
                else:
                    print(f"Warning: Mismatch in feature length for day {day}, expected 11 features.")
                    continue

            # Check window data length after the first 7 days
            #print(f"Window data length after first 7 days: {len(window_data)}")

            # Next 8-11 days: Hourly averages for each feature (4 days * 24 values per day)
            for day in range(8, 12):
                day_start = i + day * 144
                day_end = day_start + 144
                day_data = dataset_values[day_start:day_end, :]  # 每天144个数据点

                # Calculate hourly averages in a more optimized way
                # Reshape the data into (24 hours, 60 minutes, 11 features) and then calculate mean over 60 minutes
                hourly_avg = np.mean(day_data.reshape(24, 6, 11), axis=1)  # 24 hours, 60 minutes, 11 features

                # Ensure it's the correct shape and append
                if hourly_avg.shape == (24, 11):  # 24 hourly averages for each of the 11 features
                    window_data.extend(hourly_avg.flatten())
                else:
                    print(f"Warning: Mismatch in hourly averages for day {day}.")
                    continue

            # Check window data length after the hourly averages (days 8-11)
            #print(f"Window data length after hourly averages (days 8-11): {len(window_data)}")

            # Last 3 days: 10-minute interval values (3 days * 144 * 11 features)
            for day in range(12, 15):
                day_start = i + day * 144
                day_end = day_start + 144
                day_data = dataset_values[day_start:day_end, :]  # 每天144个数据点

                # Ensure the shape matches 144 intervals * 11 features and append
                if day_data.shape == (144, 11):  # 144 intervals for 11 features
                    window_data.extend(day_data.flatten())
                else:
                    print(f"Warning: Mismatch in 10-minute intervals for day {day}.")
                    continue

            # Check window data length after the 10-minute intervals (days 12-14)
            #print(f"Window data length after 10-minute intervals (days 12-14): {len(window_data)}")

            # Check the final length of window_data
            # Add target data to y after checking the conditions
            if len(window_data) == 6039:
                X.append(window_data)

                # Ensure that we are accessing the target values correctly
                target_data = dataset_values[i + window_size:i + window_size + forecast_size, -1]  # Target wind speed

                # Only append if target_data is valid and not empty
                if target_data.size == forecast_size:  # Make sure we have the correct size
                    y.append(target_data)
                else:
                    print(f"Warning: Target data size mismatch for window {len(X)}. Skipping this window.")
            else:
                print(f"Warning: Window size mismatch. Skipping this window.")
                continue

    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    print(X.shape,y.shape)

    print(f"Total valid sliding windows created with shift={shift}: {len(X)}")
    return X, y


def add_noise_tf(x, y, noise_level=0.05):
    """
    Adds Gaussian noise to input features dynamically within the TensorFlow pipeline.

    Args:
        x (tf.Tensor): Input features
        y (tf.Tensor): Target values
        noise_level (float): Standard deviation of noise

    Returns:
        x_noisy, y_noisy: Noisy features and (optionally) noisy labels
    """
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_level)
    x_noisy = x + noise  # Add noise to input features
    return x_noisy, y


def prepare_datasets(X, y, train_ratio, val_ratio, batch_size):
    """
    Memory efficient dataset preparation with proper iteration handling

    Args:
        X: Input features array
        y: Target values array
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        batch_size: Batch size for training

    Returns:
        train_dataset, val_dataset, test_dataset, X_test, y_test, dataset_info
    """
    # Calculate splits
    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size

    # Calculate steps per epoch
    steps_per_epoch = {
        'train': train_size // batch_size,
        'val': val_size // batch_size,
        'test': test_size // batch_size
    }

    # Create indices for splits
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"Training Pairs: {train_size}, Validation Pairs: {val_size}, Testing Pairs: {test_size}")

    def make_dataset(indices, is_training=False):
        """Create dataset with proper handling of iterations

        Args:
            indices: Array of indices to use for this dataset
            is_training: Whether this is for training (affects shuffling)

        Returns:
            A configured TensorFlow dataset
        """

        def generator():
            # Process in smaller chunks to save memory
            chunk_size = min(1000, len(indices))
            current_indices = indices.copy()

            # Shuffle indices if training
            if is_training:
                np.random.shuffle(current_indices)

            # Generate data in chunks
            for i in range(0, len(current_indices), chunk_size):
                chunk_indices = current_indices[i:min(i + chunk_size, len(current_indices))]
                yield (X[chunk_indices], y[chunk_indices])

        # Create dataset with proper types and shapes
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, X.shape[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None, y.shape[1]), dtype=tf.float32)
            )
        )

        # Configure dataset
        dataset = dataset.unbatch()

        # Add shuffling for training
        if is_training:
            dataset = dataset.map(lambda x,y: add_noise_tf(x,y,noise_level=0.02))
            dataset = dataset.shuffle(buffer_size=min(len(indices), 10000))


        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Return with prefetch
        return dataset.prefetch(tf.data.AUTOTUNE)

    # Create datasets with proper configuration
    train_dataset = make_dataset(train_indices, is_training=True)
    val_dataset = make_dataset(val_indices, is_training=True)
    test_dataset = make_dataset(test_indices, is_training=False)

    # Save memory by only keeping test data needed for evaluation
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Create dataset info dictionary
    # Create dataset info dictionary
    dataset_info = {
        'steps_per_epoch': steps_per_epoch,
        'sizes': {
            'train': train_size,
            'val': val_size,
            'test': test_size
        },
        'batch_size': batch_size
    }
    print("\nDataset Info:")
    print(f"Steps per epoch: {dataset_info['steps_per_epoch']}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Dataset sizes: {dataset_info['sizes']}")

    return train_dataset, val_dataset, test_dataset, X_test, y_test, dataset_info


# Optional: Add a function to calculate memory usage
def estimate_memory_usage(X, y, batch_size):
    """
    Estimates memory usage for the datasets
    """
    sample_size = (X[0].nbytes + y[0].nbytes) / (1024 * 1024)  # Size in MB
    batch_memory = sample_size * batch_size

    print(f"\nMemory Usage Estimates:")
    print(f"Single sample: {sample_size:.2f} MB")
    print(f"Batch size: {batch_size}")
    print(f"Memory per batch: {batch_memory:.2f} MB")

    return batch_memory


# Optional: Add configuration function for dataset preparation
def configure_dataset_options(dataset, batch_size, buffer_size=None):
    """
    Configures dataset options for optimal performance
    """
    if buffer_size is None:
        buffer_size = batch_size * 10

    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True

    return dataset.with_options(options)


def load_and_preprocess_data(config):
    """Data loading and preprocessing"""
    try:
        nelson_data = load_csv(config['data']['nelson_file_id'])
        greymouth_data = load_csv(config['data']['greymouth_file_id'])

        final_dataset = preprocess_data(nelson_data, greymouth_data)
        print("Data preprocessing completed!")

        X, y = create_sliding_windows(final_dataset, 2016, 144, 1)
        datasets = prepare_datasets(X, y,
                                    config['training']['train_ratio'],
                                    config['training']['val_ratio'],
                                    config['data']['batch_size'])

        # Save test dataset
        with open(config['data']['test_data_path'], "wb") as f:
            pickle.dump((datasets[3], datasets[4]), f)

        return datasets

    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        raise

