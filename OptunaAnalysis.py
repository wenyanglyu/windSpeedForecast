import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'results_dir': "optuna_results",  # Folder where the results are stored
    'result_files': ["optuna_results.pkl"],  # Files to analyze
}

# Define which parameters are structural (can't be changed during final training)
STRUCTURAL_PARAMS = ["d_model", "num_heads", "dff", "num_layers"]

# Define which parameters are non-structural (can be changed during final training)
TUNABLE_PARAMS = ["optimizer", "loss_function", "learning_rate", "dropout_rate"]


def load_optuna_trials():
    """Load Optuna trial results from pkl files."""
    all_trials = []

    for file_name in CONFIG['result_files']:
        file_path = os.path.join(CONFIG['results_dir'], file_name)

        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found. Skipping...")
            continue

        with open(file_path, "rb") as f:
            try:
                data = pickle.load(f)

                # Handle both list format and individual object format
                if isinstance(data, list):
                    all_trials.extend(data)
                else:
                    all_trials.append(data)

                print(f"Loaded {len(all_trials)} trials from {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")

    if not all_trials:
        print("No valid Optuna results found.")
        return None

    df = pd.DataFrame(all_trials)
    print(f"Total trials loaded: {len(df)}")

    return df


def analyze_best_params(df, top_n=2):
    """Analyze and find the best and second best parameters for each category."""
    if df is None or df.empty:
        print("No data available for analysis.")
        return

    # Sort by validation loss to find overall best trial
    df_sorted = df.sort_values(by="val_loss")
    best_trial = df_sorted.iloc[0].to_dict()

    print("\n===== OVERALL BEST TRIAL =====")
    for key, value in best_trial.items():
        if key != "val_loss":
            print(f"{key}: {value}")
    print(f"val_loss: {best_trial['val_loss']:.6f}")

    # Analyze structural parameters (can't change during final training)
    print("\n===== STRUCTURAL PARAMETERS (FIXED FOR FINAL MODEL) =====")
    for param in STRUCTURAL_PARAMS:
        if param in df.columns:
            # Group by parameter value and calculate mean validation loss
            param_performance = df.groupby(param)["val_loss"].agg(['mean', 'std', 'count']).reset_index()
            param_performance = param_performance.sort_values(by='mean')

            best_value = param_performance.iloc[0][param]
            best_loss = param_performance.iloc[0]['mean']

            print(f"\n{param.upper()} (Best: {best_value}, Loss: {best_loss:.6f})")
            print(f"  Top {top_n} values:")
            for i in range(min(top_n, len(param_performance))):
                value = param_performance.iloc[i][param]
                loss = param_performance.iloc[i]['mean']
                count = param_performance.iloc[i]['count']
                print(f"  {i + 1}. {param} = {value} (Loss: {loss:.6f}, Trials: {count})")

            # Plot this parameter's effect on validation loss
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[param], y=df["val_loss"])
            plt.title(f"Effect of {param} on Validation Loss")
            plt.xlabel(param)
            plt.ylabel("Validation Loss")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['results_dir'], f"{param}_impact.png"))
            plt.close()

    # Analyze tunable parameters (can change during final training)
    print("\n===== TUNABLE PARAMETERS (CAN BE ADJUSTED) =====")
    for param in TUNABLE_PARAMS:
        if param in df.columns:
            if df[param].dtype == 'float64' or df[param].dtype == 'int64':
                # For numerical parameters
                param_performance = df.groupby(param)["val_loss"].agg(['mean', 'std', 'count']).reset_index()
                param_performance = param_performance.sort_values(by='mean')
            else:
                # For categorical parameters
                param_performance = df.groupby(param)["val_loss"].agg(['mean', 'std', 'count']).reset_index()
                param_performance = param_performance.sort_values(by='mean')

            best_value = param_performance.iloc[0][param]
            best_loss = param_performance.iloc[0]['mean']

            print(f"\n{param.upper()} (Best: {best_value}, Loss: {best_loss:.6f})")
            print(f"  Top {top_n} values:")
            for i in range(min(top_n, len(param_performance))):
                value = param_performance.iloc[i][param]
                loss = param_performance.iloc[i]['mean']
                count = param_performance.iloc[i]['count']
                print(f"  {i + 1}. {param} = {value} (Loss: {loss:.6f}, Trials: {count})")

            # Create visualization
            if df[param].dtype == 'float64' or df[param].dtype == 'int64':
                # For numerical parameters: scatter plot with trend line
                plt.figure(figsize=(10, 6))
                sns.regplot(x=param, y="val_loss", data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
                plt.title(f"Effect of {param} on Validation Loss")
            else:
                # For categorical parameters: box plot
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=param, y="val_loss", data=df)
                plt.title(f"Effect of {param} on Validation Loss")

            plt.xlabel(param)
            plt.ylabel("Validation Loss")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['results_dir'], f"{param}_impact.png"))
            plt.close()


def create_param_interaction_plots(df):
    """Create plots showing interactions between parameters."""
    if df is None or df.empty or len(df) < 5:
        print("Not enough data for interaction analysis")
        return

    print("\n===== PARAMETER INTERACTIONS =====")

    # 1. Create a correlation heatmap of numerical parameters with val_loss
    numerical_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    if len(numerical_cols) > 1:  # Need at least 2 numerical columns for correlation
        plt.figure(figsize=(12, 10))
        corr = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Parameter Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['results_dir'], "parameter_correlations.png"))
        plt.close()
        print("Created parameter correlation heatmap")

    # 2. Create a pairplot for key structural parameters
    structural_cols = [col for col in STRUCTURAL_PARAMS if col in df.columns]
    if structural_cols:
        plot_cols = structural_cols + ['val_loss']
        plt.figure(figsize=(15, 12))
        sns.pairplot(df[plot_cols], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
        plt.savefig(os.path.join(CONFIG['results_dir'], "structural_params_pairplot.png"))
        plt.close()
        print("Created structural parameters pairplot")

    # 3. Create interaction plots for best performance combinations
    # Choose the top 2 most important parameters from structural and tunable
    top_structural = STRUCTURAL_PARAMS[:2]
    top_tunable = TUNABLE_PARAMS[:2]

    for s_param in top_structural:
        for t_param in top_tunable:
            if s_param in df.columns and t_param in df.columns:
                if df[t_param].dtype not in ['float64', 'int64']:  # If tunable param is categorical
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x=s_param, y='val_loss', hue=t_param, data=df)
                    plt.title(f"Interaction: {s_param} × {t_param}")
                    plt.xlabel(s_param)
                    plt.ylabel("Validation Loss")
                    plt.legend(title=t_param)
                    plt.tight_layout()
                    plt.savefig(os.path.join(CONFIG['results_dir'], f"interaction_{s_param}_{t_param}.png"))
                    plt.close()
                    print(f"Created interaction plot for {s_param} × {t_param}")


def generate_final_model_recommendation(df):
    """Generate a clear recommendation for the final model configuration."""
    if df is None or df.empty:
        print("No data available for recommendations.")
        return

    # Sort by validation loss to find best trials
    df_sorted = df.sort_values(by="val_loss")
    best_trial = df_sorted.iloc[0].to_dict()

    # For structural parameters: use the best trial's values
    structural_config = {param: best_trial[param] for param in STRUCTURAL_PARAMS if param in best_trial}

    # For tunable parameters: analyze the distribution to make recommendations
    tunable_recommendations = {}
    for param in TUNABLE_PARAMS:
        if param in df.columns:
            param_performance = df.groupby(param)["val_loss"].mean().sort_values()
            best_values = param_performance.index[:2].tolist()  # Get top 2 best values
            tunable_recommendations[param] = {
                "best": best_values[0],
                "second_best": best_values[1] if len(best_values) > 1 else None
            }

    # Generate the final recommendation
    print("\n===== FINAL MODEL RECOMMENDATION =====")
    print("\nFixed Structural Parameters:")
    for param, value in structural_config.items():
        print(f"  {param}: {value}")

    print("\nTunable Parameters (with alternatives):")
    for param, values in tunable_recommendations.items():
        print(f"  {param}: {values['best']} (alternative: {values['second_best']})")

    # Save the recommendation to a file
    with open(os.path.join(CONFIG['results_dir'], "model_recommendation.txt"), "w") as f:
        f.write("===== FINAL MODEL RECOMMENDATION =====\n\n")
        f.write("Fixed Structural Parameters:\n")
        for param, value in structural_config.items():
            f.write(f"  {param}: {value}\n")

        f.write("\nTunable Parameters (with alternatives):\n")
        for param, values in tunable_recommendations.items():
            f.write(f"  {param}: {values['best']} (alternative: {values['second_best']})\n")

    print("\nRecommendation saved to model_recommendation.txt")


def main():
    """Main function to analyze Optuna results with focus on fixed vs tunable parameters."""
    # Create results directory if it doesn't exist
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # Load trial data
    df = load_optuna_trials()

    if df is not None and not df.empty:
        # Analyze best parameters (structural vs tunable)
        analyze_best_params(df)

        # Create parameter interaction plots
        create_param_interaction_plots(df)

        # Generate final model recommendation
        generate_final_model_recommendation(df)

        print("\nAnalysis completed! Check the optuna_results directory for visualizations.")
    else:
        print("Analysis failed: No valid data loaded.")


if __name__ == "__main__":
    main()
