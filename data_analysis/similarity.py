import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import pickle
import os

CONFIG = {
    'data': {
        'test_data_path': os.path.join("..", "test_dataset.pkl"),  # ✅ Correct relative path
        'batch_size': 32
    },
    'training': {
        'epochs': 200,
        'optuna_epochs': 20,
        'n_trials': 50,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'save_path': os.path.join("..", "wind_forecast_model") # ✅ Correct relative path
    }
}


def compute_similarity_for_fold(X, y, sample_size=1000):
    """Compute pairwise similarity with memory efficiency improvements."""
    # Take a random sample if dataset is large
    if len(X) > sample_size:
        indices = np.random.choice(len(X) - 1, sample_size, replace=False)
        indices = np.sort(indices)  # Sort to maintain consecutive pairs
    else:
        indices = np.arange(len(X) - 1)

    similarity_results = {
        'cosine_sim_X': [],
        'cosine_sim_y': [],
        'euclidean_dist_X': [],
        'euclidean_dist_y': [],
        'pearson_X': [],
        'pearson_y': []
    }

    for i in indices:
        x1, x2 = X[i].flatten(), X[i + 1].flatten()
        y1, y2 = y[i].flatten(), y[i + 1].flatten()

        # Compute metrics
        similarity_results['cosine_sim_X'].append(
            cosine_similarity([x1], [x2])[0, 0])
        similarity_results['cosine_sim_y'].append(
            cosine_similarity([y1], [y2])[0, 0])
        similarity_results['euclidean_dist_X'].append(
            euclidean(x1, x2))
        similarity_results['euclidean_dist_y'].append(
            euclidean(y1, y2))
        pearson_X, _ = pearsonr(x1, x2)
        pearson_y, _ = pearsonr(y1, y2)
        similarity_results['pearson_X'].append(pearson_X)
        similarity_results['pearson_y'].append(pearson_y)

    return similarity_results


def analyze_all_splits(X_test, y_test):
    """Analyze similarities in all splits including folds."""
    print("\nAnalyzing similarities across all splits...")

    # Analyze test set
    print("\nComputing similarity metrics for Test Set...")
    test_similarities = compute_similarity_for_fold(X_test, y_test)
    print_similarity_stats("Test Set", test_similarities)
    visualize_similarity(test_similarities, "Test Set")



def print_similarity_stats(split_name, similarity_results):
    """Print similarity statistics for a specific split."""
    print(f"\nSimilarity Statistics for {split_name}:")
    for metric, values in similarity_results.items():
        values = np.array(values)
        print(f"\n{metric}:")
        print(f"  Mean: {np.nanmean(values):.4f}")
        print(f"  Median: {np.nanmedian(values):.4f}")
        print(f"  Std Dev: {np.nanstd(values):.4f}")


def visualize_similarity(similarity_results, split_name):
    """Visualize similarity metrics for a specific split."""
    plt.figure(figsize=(15, 10))

    # Cosine Similarity
    plt.subplot(2, 2, 1)
    plt.plot(similarity_results['cosine_sim_X'], label='Features', alpha=0.7)
    plt.plot(similarity_results['cosine_sim_y'], label='Targets', alpha=0.7)
    plt.xlabel("Sample Pair Index")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine Similarity ({split_name})")
    plt.legend()

    # Euclidean Distance
    plt.subplot(2, 2, 2)
    plt.plot(similarity_results['euclidean_dist_X'], label='Features', alpha=0.7)
    plt.plot(similarity_results['euclidean_dist_y'], label='Targets', alpha=0.7)
    plt.xlabel("Sample Pair Index")
    plt.ylabel("Euclidean Distance")
    plt.title(f"Euclidean Distance ({split_name})")
    plt.legend()

    # Pearson Correlation
    plt.subplot(2, 2, 3)
    plt.plot(similarity_results['pearson_X'], label='Features', alpha=0.7)
    plt.plot(similarity_results['pearson_y'], label='Targets', alpha=0.7)
    plt.xlabel("Sample Pair Index")
    plt.ylabel("Pearson Correlation")
    plt.title(f"Pearson Correlation ({split_name})")
    plt.legend()

    # Distribution plot for cosine similarity
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=similarity_results['cosine_sim_X'], label='Features', alpha=0.7)
    sns.kdeplot(data=similarity_results['cosine_sim_y'], label='Targets', alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title(f"Similarity Distribution ({split_name})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"similarity_analysis_{split_name.replace(' ', '_').lower()}.png")
    plt.close()

def load_test_dataset():
    """Load the test dataset from the pickle file."""
    try:
        with open(CONFIG['data']['test_data_path'], "rb") as f:
            test_data = pickle.load(f)
        return test_data
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        raise


def main():
    """Main function to run similarity analysis using preloaded dataset."""
    try:
        print("Loading test dataset from pickle file...")
        test_data = load_test_dataset()

        # Extract datasets
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        dataset_info = test_data.get('dataset_info', {})  # Use default empty dict if missing

        # Analyze similarities across all splits
        analyze_all_splits(X_test, y_test)

        print("\nAnalysis completed! Check the generated png files for visualizations.")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()