import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
from data_preprocessing import load_csv, preprocess_data, create_sliding_windows, prepare_datasets
from model_builder import create_transformer, load_model
from training import run_optuna_optimization, train_best_model, plot_training_progress, evaluate_model
from forecast import plot_forecast

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
        'n_trials': 20,  # Number of optuna trials
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'save_path': 'wind_forecast_model'
    },
    'paths': {
        'save_dir': '/home/em19255/fineTune',
        'model_path': "/home/em19255/fineTune/wind_forecast_model.keras"
    }
}


def print_dataset_info(dataset_info):
    """Print detailed information about the datasets"""
    print("\n📊 Dataset Information:")
    print(f"Training samples: {dataset_info['sizes']['train']:,}")
    print(f"Validation samples: {dataset_info['sizes']['val']:,}")
    print(f"Testing samples: {dataset_info['sizes']['test']:,}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Steps per epoch (training): {dataset_info['steps_per_epoch']['train']:,}")
    print(f"Steps per epoch (validation): {dataset_info['steps_per_epoch']['val']:,}")
    print(f"Steps per epoch (testing): {dataset_info['steps_per_epoch']['test']:,}")


def print_metrics(metrics, prefix=""):
    """Print evaluation metrics with proper formatting and error handling"""
    print(f"\n📊 {prefix}Evaluation Metrics:")
    try:
        # Basic metrics
        print(f"Loss: {metrics.get('loss', 'N/A'):.6f}")

        # Speed metrics
        if 'normalized_mae' in metrics:
            print(f"Normalized MAE: {metrics['normalized_mae']:.6f}")
        if 'real_speed_mae' in metrics:
            print(f"Real Speed MAE (m/s): {metrics['real_speed_mae']:.6f}")
        if 'real_speed_mse' in metrics:
            print(f"Real Speed MSE (m/s)²: {metrics['real_speed_mse']:.6f}")

        # Power metrics
        if 'power_mae' in metrics:
            print(f"Power MAE (m³/s³): {metrics['power_mae']:.2f}")
        if 'power_rmse' in metrics:
            print(f"Power RMSE (m³/s³): {metrics['power_rmse']:.2f}")

    except Exception as e:
        print(f"Error printing metrics: {str(e)}")


def train_pipeline():
    """Training pipeline including data preparation, model training and evaluation"""
    try:
        # Load and preprocess data
        nelson_data = load_csv(CONFIG['data']['nelson_file_id'])
        greymouth_data = load_csv(CONFIG['data']['greymouth_file_id'])

        final_dataset = preprocess_data(nelson_data, greymouth_data)
        X, y = create_sliding_windows(final_dataset, 2016, 144, 1)

        # Get datasets with info
        datasets = prepare_datasets(X, y,
                                    CONFIG['training']['train_ratio'],
                                    CONFIG['training']['val_ratio'],
                                    CONFIG['data']['batch_size'])
        train_dataset, val_dataset, test_dataset, X_test, y_test, dataset_info = datasets

        # Print dataset information
        print_dataset_info(dataset_info)

        # Save test dataset
        with open(CONFIG['data']['test_data_path'], "wb") as f:
            pickle.dump((X_test, y_test), f)
        print("✅ Test dataset saved")

        # Hyperparameter optimization
        best_hyperparams = run_optuna_optimization(
            train_dataset,
            val_dataset,
            CONFIG['training']['optuna_epochs'],
            CONFIG['training']['n_trials'],
            dataset_info['steps_per_epoch']  # Pass steps_per_epoch
        )
        print("✅ Hyperparameter optimization completed")

        # Before training
        print("\nBefore training:")
        print(f"Dataset info: {dataset_info}")

        # Train final model
        print("\nStarting main training...")
        model, history = train_best_model(
            best_hyperparams,
            train_dataset,
            val_dataset,
            epochs=CONFIG['training']['epochs'],
            steps_per_epoch=dataset_info['steps_per_epoch'],  # Make sure this is passed
            save_path=CONFIG['training']['save_path']
        )

        # Save model
        tf.keras.models.save_model(model, CONFIG['paths']['model_path'])
        # Plot and evaluate
        plot_training_progress(history)
        # Print and evaluate
        test_metrics = evaluate_model(model, test_dataset)
        print(f"\n📊 Final Evaluation:")
        for metric_name, value in test_metrics.items():
            print(f"{metric_name}: {value:.6f}")

        # Create initial forecast
        input_data = next(iter(test_dataset))
        forecasted_values = model.predict(input_data[0])
        time_range = range(24 * 6)
        plot_forecast(input_data[1][0], forecasted_values[0], time_range)
        print("\n✅ Initial forecast plot saved")

        return model

    except Exception as e:
        print(f"❌ Training pipeline failed: {str(e)}")
        raise


def predict_pipeline():
    """Prediction pipeline with error handling and metric printing"""
    try:
        # Load test data
        with open(CONFIG['data']['test_data_path'], "rb") as f:
            X_test, y_test = pickle.load(f)

        batch_size = CONFIG['data']['batch_size']
        total_samples = len(X_test)
        total_batches = total_samples // batch_size

        print(f"\n📊 Test Dataset Information:")
        print(f"Total test samples: {total_samples:,}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {total_batches:,}")

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        model = load_model(CONFIG['paths']['model_path'])
        test_metrics = evaluate_model(model, test_dataset)
        print_metrics(test_metrics, prefix="Test ")

        # Forecast visualization
        day_to_plot = 300
        batch_num = day_to_plot // batch_size
        index_in_batch = day_to_plot % batch_size

        for i, batch_data in enumerate(test_dataset):
            if i == batch_num:
                input_data = batch_data
                break

        forecasted_values = model.predict(input_data[0])
        time_range = range(24 * 6)
        plot_forecast(input_data[1][index_in_batch],
                      forecasted_values[index_in_batch],
                      time_range)
        print(f"\n✅ Forecast plot saved for day {day_to_plot}")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        raise


def main():
    try:
        # Check GPU availability
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            print(f"✅ Found {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        else:
            print("⚠️ No GPU found, using CPU")

        # Create necessary directories
        os.makedirs(CONFIG['paths']['save_dir'], exist_ok=True)

        MODE = 'train'  # 'train' or 'predict'

        if MODE == 'train':
            train_pipeline()
        elif MODE == 'predict':
            predict_pipeline()
        else:
            raise ValueError(f"Invalid mode: {MODE}")

    except Exception as e:
        print(f"❌ Process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()