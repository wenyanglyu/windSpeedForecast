import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import KFold

CONFIG = {
    'data': {
        'nelson_file_id': "1FftdSBsoQHrZ0dxnoYDAVYME3Gp8UXsb",
        'greymouth_file_id': "1WRW60JP8xOl0fcQIivJaUF8883k7Ii6e",
        'test_data_path': "test_dataset.pkl",
        'train_data_path': "train_data.pkl",
        'speed_column': 7,
        'batch_size': 32,
        'window_shift': 3
    },
    'training': {
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1
    }
}

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


def normalize_wind_speed(wind_speed_series):
    """
    Min-Max normalization for wind speed, handling NaN values correctly.
    Normalized range: [0,1].
    """
    # Compute min and max, ignoring NaN values
    min_speed = 0
    max_speed = 21.6
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

    # Remove timezone information
    nelson_data['OBS_DATE'] = nelson_data['OBS_DATE'].dt.tz_localize(None)
    greymouth_data['OBS_DATE'] = greymouth_data['OBS_DATE'].dt.tz_localize(None)

    # Expand Greymouth data to 10-minute intervals
    expanded_greymouth = expand_greymouth_data(greymouth_data)

    # Merge the two datasets on OBS_DATE
    combined_data = pd.merge(nelson_data, expanded_greymouth[['OBS_DATE', 'PRESSURE_MSL60']], on='OBS_DATE', how='left')

    # Convert relevant columns to numeric values
    numerical_cols = ['PRESSURE_MSL60', 'PRESSURE_MSL10', 'MEAN_TEMP10', 'MEAN_RELHUM10']
    combined_data[numerical_cols] = combined_data[numerical_cols].apply(pd.to_numeric, errors='coerce')

    # Apply forward fill to all numerical columns
    combined_data[numerical_cols] = combined_data[numerical_cols].ffill()

    # Convert MEAN_SPD10 to numeric and set invalid values to NaN
    combined_data['MEAN_SPD10'] = pd.to_numeric(combined_data['MEAN_SPD10'], errors='coerce')
    combined_data['valid_for_window'] = ~combined_data['MEAN_SPD10'].isna()

    # Normalize MEAN_SPD10
    combined_data['normalized_MEAN_SPD10'] = normalize_wind_speed(combined_data['MEAN_SPD10'])

    # Normalize pressure and other features
    for col in numerical_cols:
        combined_data[f'normalized_{col}'] = normalize_feature(combined_data[col])

    # Wind Direction Encoding (Sin and Cos)
    combined_data['MEAN_DIR10'] = pd.to_numeric(combined_data['MEAN_DIR10'], errors='coerce')
    combined_data['wind_dir_rad'] = np.radians(combined_data['MEAN_DIR10'])
    combined_data['wind_dir_sin'] = np.sin(combined_data['wind_dir_rad'])
    combined_data['wind_dir_cos'] = np.cos(combined_data['wind_dir_rad'])

    # Year Normalization
    combined_data['year'] = combined_data['OBS_DATE'].dt.year
    combined_data['normalized_year'] = (combined_data['year'] - 2015) / (2025 - 2015)

    # Day of Year Encoding (Leap Year Consideration)
    combined_data['day_of_year'] = combined_data['OBS_DATE'].dt.dayofyear
    combined_data['is_leap_year'] = combined_data['OBS_DATE'].dt.is_leap_year
    combined_data['day_of_year_sin'] = np.sin(2 * np.pi * combined_data['day_of_year'] / np.where(combined_data['is_leap_year'], 366, 365))
    combined_data['day_of_year_cos'] = np.cos(2 * np.pi * combined_data['day_of_year'] / np.where(combined_data['is_leap_year'], 366, 365))

    # Time of Day Encoding
    combined_data['minute_of_day'] = combined_data['OBS_DATE'].dt.hour * 60 + combined_data['OBS_DATE'].dt.minute

    # Print sample values before conversion
    print("\nSample values before conversion:")
    print(combined_data[['OBS_DATE', 'minute_of_day']].sample(10))

    # Apply sine and cosine transformation
    combined_data['minute_of_day_sin'] = np.sin(2 * np.pi * combined_data['minute_of_day'] / 1440)
    combined_data['minute_of_day_cos'] = np.cos(2 * np.pi * combined_data['minute_of_day'] / 1440)

    # Print sample values after conversion
    print("\nSample values after conversion:")
    print(combined_data[['minute_of_day', 'minute_of_day_sin', 'minute_of_day_cos']].sample(10))

    # Select processed features
    selected_features = [
        'normalized_year',  # Year normalization
        'day_of_year_sin', 'day_of_year_cos',  # Day encoding
        'minute_of_day_sin', 'minute_of_day_cos',  # Time encoding
        'wind_dir_sin', 'wind_dir_cos',  # Wind direction encoding
        'normalized_MEAN_SPD10'  # Normalized wind speed
    ] + [f'normalized_{col}' for col in numerical_cols] + ['valid_for_window']

    print("Selected features:", selected_features)
    print("\nFeature statistics:")
    for feature in selected_features[:-1]:  # Exclude valid_for_window
        data = combined_data[feature]
        print(f"\n{feature}:")
        print(f"Min: {data.min():.4f}")
        print(f"Max: {data.max():.4f}")
        print(f"Mean: {data.mean():.4f}")
        print(f"Std: {data.std():.4f}")

    return combined_data[selected_features]

def create_sliding_windows(wind_data, window_size=1440, forecast_size=144, shift=6, features=None):
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
                day_start = i + day * 144  # Each day has 144 time steps (10-min intervals)
                day_end = day_start + 144
                day_data = dataset_values[day_start:day_end, :]  # Extract daily data

                # Reshape into (24 hours, 6 samples per hour, 12 features)
                hourly_data = day_data.reshape(24, 6, 12)

                # Compute mean across 6 samples per hour
                hourly_avg = np.mean(hourly_data, axis=1)  # Shape: (24, 12)

                # Ensure correct shape and append
                if hourly_avg.shape == (24, 12):
                    window_data.extend(hourly_avg.flatten())  # Store all 24-hour values (24 * 12 = 288)
                else:
                    print(f"Warning: Mismatch in hourly averaging for day {day}.")
                    continue

            # Next 3 days: Use full 10-minute intervals (no averaging)
            for day in range(8, 11):
                day_start = i + day * 144
                day_end = day_start + 144
                day_data = dataset_values[day_start:day_end, :]  # Extract 144 time-step data

                # Ensure the correct shape and append
                if day_data.shape == (144, 12):
                    window_data.extend(day_data.flatten())  # Store all 144 values (144 * 12 = 1728)
                else:
                    print(f"Warning: Mismatch in 10-minute intervals for day {day}.")
                    continue

            # Check the final length of window_data
            # Add target data to y after checking the conditions
            if len(window_data) == 7200:
                X.append(window_data)

                # Ensure that we are accessing the target values correctly
                target_data = dataset_values[i + window_size:i + window_size + forecast_size, CONFIG['data']['speed_column']]  # Target wind speed

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
    print(f"Final window shapes - X: {np.array(X).shape}, Y: {np.array(y).shape}")

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


def data_generator(X, y):
    """Generator to yield one sample at a time."""
    for i in range(len(X)):
        yield X[i], y[i]


def prepare_datasets(X, y, test_ratio=0.1, val_ratio=0.1, batch_size=32, num_folds=5):
    """
    Memory-efficient dataset preparation with K-fold and controlled ratios.
    1. First separates test_ratio (10%) for final testing
    2. Then does K-fold on remaining data with val_ratio (10%) for validation
    3. Returns both TensorFlow datasets and serializable raw data
    """
    n_samples = len(X)

    # First split out test set
    test_size = int(test_ratio * n_samples)
    test_indices = np.random.choice(n_samples, test_size, replace=False)
    train_val_indices = np.setdiff1d(np.arange(n_samples), test_indices)

    # Get test data
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"\nInitial Split:")
    print(f"Total Samples: {n_samples}")
    print(f"Test Set: {test_size} samples ({test_size / n_samples:.1%})")
    print(f"Train-Val Set: {len(train_val_indices)} samples ({len(train_val_indices) / n_samples:.1%})")

    # Create test dataset
    test_dataset = tf.data.Dataset.from_generator(
        lambda: ((X[i], y[i]) for i in test_indices),
        output_signature=(
            tf.TensorSpec(shape=(X.shape[1],), dtype=tf.float32),
            tf.TensorSpec(shape=(y.shape[1],), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Prepare K-fold splits on remaining data
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_datasets = []
    fold_data_raw = []  # For serialization

    # Calculate validation size for remaining data
    remaining_samples = len(train_val_indices)
    val_size = int(val_ratio * remaining_samples)

    # Store splits for all folds
    all_fold_splits = list(kf.split(train_val_indices))

    for fold, (train_idx, val_idx) in enumerate(all_fold_splits):
        # Ensure validation set is exactly val_ratio
        if len(val_idx) > val_size:
            # Move excess samples to training
            move_to_train = val_idx[val_size:]
            val_idx = val_idx[:val_size]
            train_idx = np.concatenate([train_idx, move_to_train])

        # Get actual indices
        train_indices = train_val_indices[train_idx]
        val_indices = train_val_indices[val_idx]

        print(f"\nFold {fold + 1}:")
        print(f"Train size: {len(train_indices)} samples ({len(train_indices) / remaining_samples:.1%})")
        print(f"Val size: {len(val_indices)} samples ({len(val_indices) / remaining_samples:.1%})")

        # Create memory-efficient datasets
        train_dataset = tf.data.Dataset.from_generator(
            lambda: ((X[i], y[i]) for i in train_indices),
            output_signature=(
                tf.TensorSpec(shape=(X.shape[1],), dtype=tf.float32),
                tf.TensorSpec(shape=(y.shape[1],), dtype=tf.float32)
            )
        ).shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            lambda: ((X[i], y[i]) for i in val_indices),
            output_signature=(
                tf.TensorSpec(shape=(X.shape[1],), dtype=tf.float32),
                tf.TensorSpec(shape=(y.shape[1],), dtype=tf.float32)
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Apply performance optimizations
        options = tf.data.Options()
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_parallelization = True

        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

        # Add to TensorFlow datasets list
        fold_datasets.append((train_dataset, val_dataset))

        # Extract raw arrays for serialization
        # Get the actual data for this fold
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Add to serializable data list
        fold_data_raw.append({
            'train_X': X_train,
            'train_y': y_train,
            'val_X': X_val,
            'val_y': y_val
        })

    # Calculate dataset info
    dataset_info = {
        'batch_size': batch_size,
        'num_folds': num_folds,
        'steps_per_epoch': {
            'train': len(train_indices) // batch_size,
            'val': len(val_indices) // batch_size,
            'test': test_size // batch_size
        },
        'sizes': {
            'total': n_samples,
            'train': len(train_indices),
            'val': len(val_indices),
            'test': test_size
        },
        'ratios': {
            'test': test_size / n_samples,
            'train': len(train_indices) / remaining_samples,
            'val': len(val_indices) / remaining_samples
        }
    }

    for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
        for x_batch, y_batch in train_dataset.take(1):
            print(f"Fold {fold_idx} - X batch shape: {x_batch.shape}, Y batch shape: {y_batch.shape}")

    # Save fold_datasets.pkl with raw data
    print("Saving fold_datasets.pkl...")
    with open("fold_datasets.pkl", "wb") as f:
        pickle.dump((fold_data_raw, dataset_info), f)
    print(f"Saved fold_datasets.pkl with {len(fold_data_raw)} folds")

    return fold_datasets, test_dataset, X_test, y_test, dataset_info


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


def load_and_preprocess_data(config, k_folds=5):
    """Data loading and preprocessing with K-Fold Cross-Validation."""
    try:
        # Load and preprocess data
        nelson_data = load_csv(config['data']['nelson_file_id'])
        greymouth_data = load_csv(config['data']['greymouth_file_id'])
        final_dataset = preprocess_data(nelson_data, greymouth_data)

        # Create sliding windows
        X, y = create_sliding_windows(final_dataset, 1440, 144, 6)

        # Split test set first
        n_samples = len(X)
        test_size = int(0.1 * n_samples)
        test_indices = np.random.choice(n_samples, test_size, replace=False)
        train_val_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        # Split data
        X_test, y_test = X[test_indices], y[test_indices]
        X_train_val, y_train_val = X[train_val_indices], y[train_val_indices]

        # Save test dataset
        with open(config['data']['test_data_path'], "wb") as f:
            pickle.dump((X_test, y_test), f)

        # Prepare K-fold datasets
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_datasets = []

        print(f"Creating {k_folds} folds with KFold...")
        print(f"X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
            print(f"Processing fold {fold + 1}...")
            print(f"Train indices: {len(train_idx)}, Val indices: {len(val_idx)}")

            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

            # Create TensorFlow datasets
            print(f"Creating TensorFlow datasets for fold {fold + 1}...")
            try:
                train_dataset = tf.data.Dataset.from_generator(
                    lambda: data_generator(X_train, y_train),
                    output_signature=(
                        tf.TensorSpec(shape=(X_train.shape[1],), dtype=tf.float32),
                        tf.TensorSpec(shape=(y_train.shape[1],), dtype=tf.float32)
                    )
                ).batch(config['data']['batch_size']).prefetch(tf.data.AUTOTUNE)

                print(f"Train dataset created successfully for fold {fold + 1}")

                val_dataset = tf.data.Dataset.from_generator(
                    lambda: data_generator(X_val, y_val),
                    output_signature=(
                        tf.TensorSpec(shape=(X_val.shape[1],), dtype=tf.float32),
                        tf.TensorSpec(shape=(y_val.shape[1],), dtype=tf.float32)
                    )
                ).batch(config['data']['batch_size']).prefetch(tf.data.AUTOTUNE)

                print(f"Val dataset created successfully for fold {fold + 1}")

                # Apply dataset options
                train_dataset = configure_dataset_options(train_dataset, config['data']['batch_size'])
                val_dataset = configure_dataset_options(val_dataset, config['data']['batch_size'])

                print(f"Dataset options applied for fold {fold + 1}")

                fold_datasets.append((train_dataset, val_dataset))
                print(f"Fold {fold + 1} added to fold_datasets")

            except Exception as e:
                print(f"Error creating datasets for fold {fold + 1}: {str(e)}")
                raise

            print(f"Fold {fold + 1}: Train size = {len(X_train)}, Val size = {len(X_val)}")

        print(f"Created {len(fold_datasets)} fold datasets")

        # Save fold_datasets in a format that can be pickled
        print("Preparing fold_datasets for saving...")
        fold_data_raw = []

        try:
            for i, (train_ds, val_ds) in enumerate(fold_datasets):
                print(f"Preparing fold {i + 1} for saving...")

                # Create a dictionary to hold fold data
                fold_dict = {
                    'train_X': X_train_val[kf.split(X_train_val)[i][0]],
                    'train_y': y_train_val[kf.split(X_train_val)[i][0]],
                    'val_X': X_train_val[kf.split(X_train_val)[i][1]],
                    'val_y': y_train_val[kf.split(X_train_val)[i][1]]
                }
                fold_data_raw.append(fold_dict)
                print(f"Fold {i + 1} prepared for saving")

            # Save fold_datasets.pkl with raw NumPy arrays
            with open("fold_datasets.pkl", "wb") as f:
                pickle.dump((fold_data_raw, dataset_info), f)
            print(f"Saved fold_datasets.pkl with {len(fold_data_raw)} folds")

        except Exception as e:
            print(f"Error saving fold_datasets: {str(e)}")
            raise

        return fold_datasets, (X_test, y_test)

    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        raise


def print_dataset_info(dataset_info):
    """Print detailed information about the datasets"""
    print("\n Dataset Information:")
    print(f"Training samples: {dataset_info['sizes']['train']:,}")
    print(f"Validation samples: {dataset_info['sizes']['val']:,}")
    print(f"Testing samples: {dataset_info['sizes']['test']:,}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Steps per epoch (training): {dataset_info['steps_per_epoch']['train']:,}")
    print(f"Steps per epoch (validation): {dataset_info['steps_per_epoch']['val']:,}")
    print(f"Steps per epoch (testing): {dataset_info['steps_per_epoch']['test']:,}")


def create_tf_dataset(X, y, batch_size):
    """Create TF dataset from numpy arrays"""
    return tf.data.Dataset.from_tensor_slices((X, y)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)


def main():
    try:
        # Load and preprocess data
        nelson_data = load_csv(CONFIG['data']['nelson_file_id'])
        greymouth_data = load_csv(CONFIG['data']['greymouth_file_id'])

        final_dataset = preprocess_data(nelson_data, greymouth_data)
        window_shift = CONFIG['data']['window_shift']
        X, y = create_sliding_windows(final_dataset, 1440, 144, window_shift)

        # Split data into test and training sets
        n_samples = len(X)
        test_size = int(CONFIG['training']['test_ratio'] * n_samples)
        test_indices = np.random.choice(n_samples, test_size, replace=False)
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        # Get test and training data
        X_test, y_test = X[test_indices], y[test_indices]
        X_train, y_train = X[train_indices], y[train_indices]

        print(f"\nData Split:")
        print(f"Total Samples: {n_samples}")
        print(f"Test Set: {len(X_test)} samples ({len(X_test) / n_samples:.1%})")
        print(f"Training Set: {len(X_train)} samples ({len(X_train) / n_samples:.1%})")

        # Calculate dataset info
        batch_size = CONFIG['data']['batch_size']
        dataset_info = {
            'batch_size': batch_size,
            'steps_per_epoch': {
                'train': len(X_train) // batch_size,
                'val': int(len(X_train) * 0.1) // batch_size,  # Approx validation size
                'test': len(X_test) // batch_size
            },
            'sizes': {
                'total': n_samples,
                'train': len(X_train),
                'test': len(X_test)
            }
        }

        # Save test data
        test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'dataset_info': dataset_info
        }
        with open(CONFIG['data']['test_data_path'], "wb") as f:
            pickle.dump(test_data, f)
        print("Test dataset saved")

        # Save training data for fold creation in training.py
        train_data = {
            'X_train': X_train,
            'y_train': y_train,
            'dataset_info': dataset_info
        }
        with open("fold_datasets.pkl", "wb") as f:
            pickle.dump(train_data, f)
        print("Training data saved for fold creation")

    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
