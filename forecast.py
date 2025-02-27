import importlib.util
import sys
import os
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from model_builder import TemporalEmbedding, create_transformer

# Configuration (mirroring the training script's config)
CONFIG = {
    'data': {
        'test_data_path': "test_dataset.pkl",
        'model_path': "wind_forecast_model.keras"
    }
}


def load_test_dataset():
    """Load the test dataset from pickle file"""
    with open(CONFIG['data']['test_data_path'], "rb") as f:
        test_data = pickle.load(f)
    return test_data


def denormalize_wind_speed(normalized_speed):
    """Denormalize wind speed values"""
    min_speed = 0.0
    max_speed = 21.6
    return normalized_speed * (max_speed - min_speed) + min_speed


def calculate_power_output(wind_speed, air_density=1.225):
    """
    Calculate power output using wind power equation
    Power = 0.5 * ρ * A * v³
    Assumptions:
    - Air density (ρ): Standard 1.225 kg/m³
    - Rotor area (A): Assume 1 m² (can be adjusted if needed)
    - Swept area coefficient: Simplified calculation
    """
    # Cut-in speed (typical for wind turbines)
    cut_in_speed = 3.0  # m/s
    # Rated speed
    rated_speed = 12.0  # m/s
    # Cut-out speed
    cut_out_speed = 20.0  # m/s

    # Check wind speed limits
    if wind_speed < cut_in_speed or wind_speed > cut_out_speed:
        return 0.0

    # Simple power curve approximation
    if wind_speed < rated_speed:
        # Cubic power relationship below rated speed
        power = 0.5 * air_density * 1 * (wind_speed ** 3)
    else:
        # Constant power at rated speed
        power = 0.5 * air_density * 1 * (rated_speed ** 3)

    return power


def find_lowest_temperature_indices(test_data, n=5):
    """
    Find indices of n lowest temperature values in the 10th column
    """
    # Extract temperature column (index 9 since Python uses 0-based indexing)
    temperatures = test_data['X_test'][:, 9]

    # Find indices of n lowest temperatures
    lowest_temp_indices = np.argsort(temperatures)[:n]

    return lowest_temp_indices



def compare_power_output(test_data, model):
    """
    Compare predicted and actual power output for lowest temperature days
    """
    try:
        # Create results directory
        results_dir = 'forecast_results'
        os.makedirs(results_dir, exist_ok=True)

        # Prepare output log file
        log_file_path = os.path.join(results_dir, 'forecast_metrics.txt')

        # Redirect print output to both console and file
        import sys
        class Tee:
            def __init__(self, filename, stream):
                self.file = open(filename, 'w')
                self.stream = stream

            def write(self, data):
                self.file.write(data)
                self.stream.write(data)

            def flush(self):
                self.file.flush()
                self.stream.flush()

            def close(self):
                self.file.close()

        # Redirect stdout
        stdout = sys.stdout
        sys.stdout = Tee(log_file_path, stdout)

        # Find indices of lowest temperature days
        lowest_temp_indices = find_lowest_temperature_indices(test_data)

        # Select input features and actual wind speeds for these days
        X_lowest_temp = test_data['X_test'][lowest_temp_indices]
        y_actual = test_data['y_test'][lowest_temp_indices]

        # Predict wind speeds
        y_predicted = model.predict(X_lowest_temp)

        # Ensure consistent shapes
        if y_predicted.ndim == 3 and y_predicted.shape[-1] == 1:
            y_predicted = y_predicted.squeeze(-1)

        # Denormalize actual and predicted wind speeds
        y_actual_denorm = denormalize_wind_speed(y_actual)
        y_predicted_denorm = denormalize_wind_speed(y_predicted)

        # Calculate power output for actual and predicted wind speeds
        actual_power_output = np.array([
            [calculate_power_output(speed) for speed in day_speeds]
            for day_speeds in y_actual_denorm
        ])

        predicted_power_output = np.array([
            [calculate_power_output(speed) for speed in day_speeds]
            for day_speeds in y_predicted_denorm
        ])

        # Determine plot layout
        num_days = len(lowest_temp_indices)
        fig, axs = plt.subplots(num_days, 2, figsize=(20, 7 * num_days))

        # Flatten axs if only one day to ensure consistent indexing
        if num_days == 1:
            axs = axs.reshape(1, -1)

        # Create subplots for each lowest temperature day
        for i in range(num_days):
            # Wind Speed Subplot
            axs[i, 0].plot(y_actual_denorm[i], 'b--', label='Actual Wind Speed')
            axs[i, 0].plot(y_predicted_denorm[i], 'r-', label='Predicted Wind Speed')
            axs[i, 0].set_title(f'Day {lowest_temp_indices[i]} - Wind Speed')
            axs[i, 0].set_xlabel('Time Interval')
            axs[i, 0].set_ylabel('Wind Speed (m/s)')
            axs[i, 0].legend()
            axs[i, 0].grid(True)

            # Power Output Subplot
            axs[i, 1].plot(actual_power_output[i], 'b--', label='Actual Power')
            axs[i, 1].plot(predicted_power_output[i], 'r-', label='Predicted Power')
            axs[i, 1].set_title(f'Day {lowest_temp_indices[i]} - Power Output')
            axs[i, 1].set_xlabel('Time Interval')
            axs[i, 1].set_ylabel('Power (W)')
            axs[i, 1].legend()
            axs[i, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'wind_speed_power_comparison.png'), dpi=300)
        plt.close()

        # Calculate and print comparison metrics
        print("\nWind Speed and Power Output Comparison:")
        for i, day_idx in enumerate(lowest_temp_indices):
            # Wind Speed Metrics
            actual_day_speeds = y_actual_denorm[i]
            predicted_day_speeds = y_predicted_denorm[i]

            wind_speed_mae = np.mean(np.abs(actual_day_speeds - predicted_day_speeds))
            wind_speed_rmse = np.sqrt(np.mean((actual_day_speeds - predicted_day_speeds) ** 2))
            wind_speed_mape = np.mean(np.abs((actual_day_speeds - predicted_day_speeds) / actual_day_speeds)) * 100

            # Power Output Metrics
            actual_day_power = np.mean(actual_power_output[i])
            predicted_day_power = np.mean(predicted_power_output[i])

            print(f"\nDay {day_idx}:")

            # Wind Speed Metrics
            print("\nWind Speed Metrics:")
            print(f"  Actual Avg Wind Speed: {np.mean(actual_day_speeds):.2f} m/s")
            print(f"  Predicted Avg Wind Speed: {np.mean(predicted_day_speeds):.2f} m/s")
            print(f"  Mean Absolute Error (MAE): {wind_speed_mae:.2f} m/s")
            print(f"  Root Mean Square Error (RMSE): {wind_speed_rmse:.2f} m/s")
            print(f"  Mean Absolute Percentage Error (MAPE): {wind_speed_mape:.2f}%")

            # Power Output Metrics
            print("\nPower Output Metrics:")
            print(f"  Actual Avg Power Output: {actual_day_power:.2f} W")
            print(f"  Predicted Avg Power Output: {predicted_day_power:.2f} W")
            print(f"  Absolute Power Error: {abs(actual_day_power - predicted_day_power):.2f} W")
            print(
                f"  Relative Power Error: {abs(actual_day_power - predicted_day_power) / actual_day_power * 100:.2f}%")

            # Save individual day plots
            day_fig, day_axs = plt.subplots(1, 2, figsize=(15, 6))

            # Wind Speed Subplot
            day_axs[0].plot(actual_day_speeds, 'b--', label='Actual Wind Speed')
            day_axs[0].plot(predicted_day_speeds, 'r-', label='Predicted Wind Speed')
            day_axs[0].set_title(f'Day {day_idx} - Wind Speed')
            day_axs[0].set_xlabel('Time Interval')
            day_axs[0].set_ylabel('Wind Speed (m/s)')
            day_axs[0].legend()
            day_axs[0].grid(True)

            # Power Output Subplot
            day_axs[1].plot(actual_power_output[i], 'b--', label='Actual Power')
            day_axs[1].plot(predicted_power_output[i], 'r-', label='Predicted Power')
            day_axs[1].set_title(f'Day {day_idx} - Power Output')
            day_axs[1].set_xlabel('Time Interval')
            day_axs[1].set_ylabel('Power (W)')
            day_axs[1].legend()
            day_axs[1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'day_{day_idx}_comparison.png'), dpi=300)
            plt.close(day_fig)

        # Restore stdout
        sys.stdout = stdout

    except Exception as e:
        print(f"Error in compare_power_output: {e}")
        import traceback
        traceback.print_exc()

def calculate_average_metrics(test_data, model):
    """
    Calculate the average metrics (MAE, MSE, Power MAE, Power MSE) across the entire test dataset.
    """
    try:
        # Prepare the actual and predicted wind speeds
        X_test = test_data['X_test']
        y_actual = test_data['y_test']

        # Predict wind speeds for the entire test set
        y_predicted = model.predict(X_test)

        # Ensure consistent shapes
        if y_predicted.ndim == 3 and y_predicted.shape[-1] == 1:
            y_predicted = y_predicted.squeeze(-1)

        # Denormalize actual and predicted wind speeds
        y_actual_denorm = denormalize_wind_speed(y_actual)
        y_predicted_denorm = denormalize_wind_speed(y_predicted)

        # Calculate power output for actual and predicted wind speeds
        actual_power_output = np.array([
            [calculate_power_output(speed) for speed in day_speeds]
            for day_speeds in y_actual_denorm
        ])

        predicted_power_output = np.array([
            [calculate_power_output(speed) for speed in day_speeds]
            for day_speeds in y_predicted_denorm
        ])

        # Calculate metrics across all days in the test dataset
        # Wind speed MAE and MSE
        wind_speed_mae = np.mean(np.abs(y_actual_denorm - y_predicted_denorm))
        wind_speed_mse = np.mean((y_actual_denorm - y_predicted_denorm) ** 2)

        # Power MAE and MSE
        power_mae = np.mean(np.abs(actual_power_output - predicted_power_output))
        power_mse = np.mean((actual_power_output - predicted_power_output) ** 2)

        # Print the average metrics
        print("\nAverage Metrics Across Entire Test Dataset:")
        print(f"Average Wind Speed MAE: {wind_speed_mae:.2f} m/s")
        print(f"Average Wind Speed MSE: {wind_speed_mse:.2f} m²/s²")
        print(f"Average Power MAE: {power_mae:.2f} W")
        print(f"Average Power MSE: {power_mse:.2f} W²")

    except Exception as e:
        print(f"Error in calculate_average_metrics: {e}")
        import traceback
        traceback.print_exc()


def import_module_from_path(module_name, module_path):
    """
    Dynamically import a module from a specific file path
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_custom_model(model_path):
    """
    Load model with comprehensive custom objects handling
    """
    try:
        # Import necessary modules
        from model_builder import (
            TemporalEmbedding as OriginalTemporalEmbedding,
            combined_loss,
            power_loss
        )
        from training import (
            real_speed_mae,
            real_speed_mse,
            power_mae,
            power_rmse
        )

        # Create a more flexible TemporalEmbedding wrapper
        class FlexibleTemporalEmbedding(OriginalTemporalEmbedding):
            def __init__(self, d_model, **kwargs):
                # Remove unexpected arguments
                kwargs.pop('trainable', None)
                kwargs.pop('dtype', None)
                super().__init__(d_model)

            def get_config(self):
                config = super().get_config()
                config['d_model'] = self.d_model
                return config

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        # Custom objects dictionary
        custom_objects = {
            'TemporalEmbedding': FlexibleTemporalEmbedding,
            'combined_loss': combined_loss,
            'power_loss': power_loss,
            'real_speed_mae': real_speed_mae,
            'real_speed_mse': real_speed_mse,
            'power_mae': power_mae,
            'power_rmse': power_rmse
        }

        # Load model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

        # Recompile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.Huber(),
            metrics=[
                'mae',
                'mse',
                real_speed_mae,
                real_speed_mse,
                power_mae,
                power_rmse
            ]
        )

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Load test dataset
    test_data = load_test_dataset()

    # Load the saved model with custom handling
    model = load_custom_model(CONFIG['data']['model_path'])

    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Compare power output for lowest temperature days
    compare_power_output(test_data, model)

    # Calculate and print average metrics across the entire dataset
    calculate_average_metrics(test_data, model)



if __name__ == "__main__":
    main()
