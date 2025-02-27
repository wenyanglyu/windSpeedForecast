import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

# Define correct feature names
FEATURE_NAMES = [
    'normalized_year',
    'day_of_year_sin', 'day_of_year_cos',
    'minute_of_day_sin', 'minute_of_day_cos',
    'wind_dir_sin', 'wind_dir_cos',
    'normalized_MEAN_SPD10',
    'normalized_PRESSURE_MSL60',
    'normalized_PRESSURE_MSL10',
    'normalized_MEAN_TEMP10',
    'normalized_MEAN_RELHUM10'
]


CONFIG = {
    'data': {
        'nelson_file_id': "1FftdSBsoQHrZ0dxnoYDAVYME3Gp8UXsb",
        'greymouth_file_id': "1WRW60JP8xOl0fcQIivJaUF8883k7Ii6e",
        'test_data_path': "test_dataset.pkl",
        'batch_size': 32
    },
    'training': {
        'epochs': 200,  # Main training epochs
        'optuna_epochs': 20,  # Epochs for each optuna trial
        'n_trials': 50,  # Number of optuna trials
        'train_ratio': 0.889,
        'val_ratio': 0.111,
        'save_path': 'wind_forecast_model'
    },
    'paths': {
        'save_dir': '/home/em19255/fineTune',
        'model_path': "/home/em19255/fineTune/wind_forecast_model.keras"
    }
}

def check_nan_values(X, feature_names=FEATURE_NAMES):
    """Check for NaN values in each feature"""
    X_features = X[:, :12]  # First complete feature set

    print("\nNaN Value Analysis:")
    print("===================")
    for i, feature in enumerate(feature_names):
        nan_count = np.isnan(X_features[:, i]).sum()
        total_count = X_features.shape[0]
        nan_percentage = (nan_count / total_count) * 100
        print(f"{feature:25}: {nan_count:6d} NaN values ({nan_percentage:.2f}%)")

    return X_features

def clean_data_for_analysis(X, feature_names=FEATURE_NAMES):
    """Remove rows with NaN values"""
    X_features = X[:, :12]  # First complete feature set

    # Find rows without NaN values
    valid_rows = ~np.isnan(X_features).any(axis=1)
    X_clean = X_features[valid_rows]

    print(f"\nOriginal shape: {X_features.shape}")
    print(f"Shape after removing NaN values: {X_clean.shape}")

    return X_clean

def analyze_feature_correlations(X_clean, feature_names=FEATURE_NAMES):
    """Analyze correlations between features with updated feature set."""
    print("\nDebug information before correlation calculation:")
    print("Shape of X_clean:", X_clean.shape)

    # Handle NaNs in correlation calculation
    corr_matrix = np.zeros((len(feature_names), len(feature_names)))
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            valid_idx = ~(np.isnan(X_clean[:, i]) | np.isnan(X_clean[:, j]))
            if np.sum(valid_idx) > 0:
                corr_matrix[i, j] = np.corrcoef(X_clean[valid_idx, i],
                                                X_clean[valid_idx, j])[0, 1]
            else:
                corr_matrix[i, j] = np.nan

    # Create correlation DataFrame
    corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)

    # Plot correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix (Updated Features)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()

    return corr_df

def analyze_feature_importance(X_clean, y, feature_names=FEATURE_NAMES):
    """Analyze feature importance relative to wind speed prediction using mutual information."""
    y_target = y[~np.isnan(X_clean).any(axis=1)][:, 0]

    # Compute mutual information scores
    mi_scores = mutual_info_regression(X_clean, y_target)

    # Create importance DataFrame
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mi_scores})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance for Wind Speed Prediction (Updated Features)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return importance_df

def debug_nan_values(X, feature_names=FEATURE_NAMES):
    """Detailed debug of NaN values"""
    X_features = X[:, :12]  # First complete feature set

    print("\nDetailed NaN Analysis:")
    print("=====================")
    for i, feature in enumerate(feature_names):
        feature_data = X_features[:, i]

        # Basic statistics
        print(f"\nFeature: {feature}")
        print(f"Min: {np.nanmin(feature_data):.4f}")
        print(f"Max: {np.nanmax(feature_data):.4f}")
        print(f"Mean: {np.nanmean(feature_data):.4f}")
        print(f"Std: {np.nanstd(feature_data):.4f}")

        # NaN analysis
        nan_count = np.isnan(feature_data).sum()
        inf_count = np.isinf(feature_data).sum()
        zero_count = (feature_data == 0).sum()

        print(f"NaN count: {nan_count}")
        print(f"Inf count: {inf_count}")
        print(f"Zero count: {zero_count}")

        # Print some non-NaN values
        valid_values = feature_data[~np.isnan(feature_data)]
        if len(valid_values) > 0:
            print(f"First 5 valid values: {valid_values[:5]}")
        else:
            print("No valid values found!")

        # If there are NaNs, print surrounding values
        if nan_count > 0:
            nan_indices = np.where(np.isnan(feature_data))[0]
            print(f"\nFirst NaN occurrence at index: {nan_indices[0]}")
            start_idx = max(0, nan_indices[0] - 2)
            end_idx = min(len(feature_data), nan_indices[0] + 3)
            print(f"Values around first NaN:")
            for idx in range(start_idx, end_idx):
                print(f"Index {idx}: {feature_data[idx]}")

def analyze_windspeed_relationships(X_clean, y, feature_names=FEATURE_NAMES):
    """
    Analyze relationships between input features (X_clean) and the true wind speed target (y).
    Assumes y has shape (N, 1).
    """
    # The real target is in y[:, 0]
    wind_speed = y[:, 0]

    # Create scatter plots for each feature vs the true wind speed
    n_features = len(feature_names)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()

    for i in range(n_features):
        ax = axes[i]
        feature_vals = X_clean[:, i]

        # Scatter plot: feature_vals vs. true wind speed
        ax.scatter(feature_vals, wind_speed, alpha=0.1, s=1)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Normalized Wind Speed')
        ax.set_title(f'{feature_names[i]} vs Wind Speed')

    plt.tight_layout()
    plt.savefig('feature_relationships.png')
    plt.close()



def analyze_target_distribution(y):
    """Analyze the distribution of target wind speed values"""
    plt.figure(figsize=(10, 6))
    plt.hist(y[:, 0], bins=50, density=True)
    plt.title('Distribution of Target Wind Speed Values')
    plt.xlabel('Normalized Wind Speed')
    plt.ylabel('Density')
    plt.savefig('target_distribution.png')
    plt.close()

def print_dataset_info(dataset_info):
    print("\nDataset Info:")
    print(f"Steps per epoch: {dataset_info['steps_per_epoch']}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Dataset sizes: {dataset_info['sizes']}")

def main():
    """Runs the analysis pipeline using preloaded test dataset."""
    print("Loading test dataset...")

    # Load test dataset from pickle file
    test_data_path = CONFIG['data']['test_data_path']
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    # Extract features and target values
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    # If dataset_info exists, print dataset details
    dataset_info = test_data.get('dataset_info', None)
    if dataset_info:
        print_dataset_info(dataset_info)

    # Check for NaN values
    X_features = check_nan_values(X_test)

    # Clean data for analysis
    X_clean = clean_data_for_analysis(X_test)

    # Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    corr_df = analyze_feature_correlations(X_clean)

    # Print correlation summary
    wind_speed_correlations = corr_df['normalized_MEAN_SPD10'].sort_values(ascending=False)
    print("\nCorrelations with Wind Speed (after removing NaN values):")
    for feat, corr in wind_speed_correlations.items():
        if feat != 'normalized_MEAN_SPD10':
            print(f"{feat:25}: {corr:6.3f}")

    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(X_clean, y_test)
    print("\nFeature Importance Rankings:")
    print(importance_df)

    # Analyze relationships with wind speed
    print("\nAnalyzing relationships with wind speed...")
    analyze_windspeed_relationships(X_clean, y_test)

    print("\nAnalysis completed. Check the generated .png files for visualizations.")


if __name__ == "__main__":
    main()