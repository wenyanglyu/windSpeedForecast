import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model_builder import create_transformer, combined_loss, power_loss

# Configuration
CONFIG = {
    'data': {
        'fold_datasets_path': "fold_datasets.pkl",
        'test_data_path': "test_dataset.pkl"
    },
    'training': {
        'epochs': 200,
        'save_path': 'wind_forecast_model',
        'batch_size': 32,
        'optimizer_patience': 10,  # Switch optimizer after 5 epochs without improvement
        'early_stop_patience': 25  # Stop training if no improvement for 15 epochs
    },
    'paths': {
        'model_path': "wind_forecast_model.keras",
        'logs_dir': 'logs'
    },
    'model': {
        # Best hyperparameters from analysis
        'd_model': 320,  # Using top performer from group analysis
        'num_heads': 16,  # Using top performer from group analysis
        'dff': 768,  # Using top performer from group analysis
        'num_layers': 2,  # Using top performer from group analysis
        'dropout_rate': 0.0521  # From best trial
    },
    'optimizers': {
        'primary': 'adam',  # Start with the best performer
        'secondary': 'adamw'  # Switch to this if plateaued
    }
}


# Custom metrics for real wind speed and power
def denorm_tf(normalized_wind_speed):
    """Denormalize wind speed values, handling both numpy arrays and tensors."""
    min_speed = 0.0
    max_speed = 21.6

    # Handle tensor inputs without calling numpy()
    if tf.is_tensor(normalized_wind_speed):
        # Use tf operations for tensor inputs (during graph building)
        denormalized = normalized_wind_speed * (max_speed - min_speed) + min_speed
    else:
        # Handle numpy arrays or Python scalars
        denormalized = normalized_wind_speed * (max_speed - min_speed) + min_speed

    return denormalized


def real_speed_mae(y_true, y_pred):
    """Calculate MAE using denormalized wind speeds"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    # Use TF operations directly for denormalization
    min_speed = 0.0
    max_speed = 21.6
    y_true_denorm = y_true * (max_speed - min_speed) + min_speed
    y_pred_denorm = y_pred * (max_speed - min_speed) + min_speed

    return tf.reduce_mean(tf.abs(y_true_denorm - y_pred_denorm))


def real_speed_mse(y_true, y_pred):
    """Calculate MSE using denormalized wind speeds"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    # Use TF operations directly for denormalization
    min_speed = 0.0
    max_speed = 21.6
    y_true_denorm = y_true * (max_speed - min_speed) + min_speed
    y_pred_denorm = y_pred * (max_speed - min_speed) + min_speed

    return tf.reduce_mean(tf.square(y_true_denorm - y_pred_denorm))


def power_mae(y_true, y_pred):
    """Calculate MAE of wind power (V³)"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    # Use TF operations directly for denormalization
    min_speed = 0.0
    max_speed = 21.6
    y_true_denorm = y_true * (max_speed - min_speed) + min_speed
    y_pred_denorm = y_pred * (max_speed - min_speed) + min_speed

    y_true_power = tf.pow(y_true_denorm, 3)
    y_pred_power = tf.pow(y_pred_denorm, 3)
    return tf.reduce_mean(tf.abs(y_true_power - y_pred_power))


def power_rmse(y_true, y_pred):
    """Calculate RMSE of wind power (V³)"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    # Use TF operations directly for denormalization
    min_speed = 0.0
    max_speed = 21.6
    y_true_denorm = y_true * (max_speed - min_speed) + min_speed
    y_pred_denorm = y_pred * (max_speed - min_speed) + min_speed

    y_true_power = tf.pow(y_true_denorm, 3)
    y_pred_power = tf.pow(y_pred_denorm, 3)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true_power - y_pred_power)))


class AdvancedTrainingCallback(tf.keras.callbacks.Callback):
    """
    Advanced callback that alternates between optimizers and increases learning rate
    when validation loss plateaus.
    """

    def __init__(self, patience=5, verbose=1):
        super(AdvancedTrainingCallback, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best_val_loss = float('inf')

        # Training state
        self.current_optimizer = 'adam'  # Start with adam
        self.base_learning_rate = 0.001  # Initial learning rate
        self.current_lr = self.base_learning_rate
        self.switch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')

        # Check if validation loss improved
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1

            # If patience exceeded, make changes
            if self.wait >= self.patience:
                self.switch_count += 1

                # Decide what to change (optimizer or learning rate)
                if self.switch_count % 2 == 1:
                    # First switch - just change optimizer
                    new_optimizer = 'adamw' if self.current_optimizer == 'adam' else 'adam'
                    new_lr = self.current_lr

                    if self.verbose:
                        print(f"\n\nSwitching optimizer from {self.current_optimizer} to {new_optimizer}")
                        print(f"Keeping learning rate at {new_lr}")
                else:
                    # Second switch - change back to first optimizer and adjust learning rate
                    new_optimizer = 'adam' if self.current_optimizer == 'adamw' else 'adamw'
                    optimal_lr = 0.0007  # Optuna's best learning rate

                    # Adjust learning rate based on current value
                    if abs(self.current_lr - optimal_lr) < 1e-6:  # If current LR is already at optimal
                        new_lr = min(self.current_lr * 2.0, 0.01)  # Try higher but cap at 0.01
                    elif self.current_lr > optimal_lr:
                        new_lr = optimal_lr  # Go back to optimal
                    else:  # current_lr < optimal_lr
                        new_lr = optimal_lr  # Go up to optimal

                    if self.verbose:
                        print(f"\n\nSwitching optimizer from {self.current_optimizer} to {new_optimizer}")
                        print(f"Adjusting learning rate from {self.current_lr} to {new_lr}")

                # Create and apply the new optimizer
                if new_optimizer == 'adam':
                    self.model.optimizer = tf.keras.optimizers.Adam(learning_rate=new_lr)
                else:
                    self.model.optimizer = tf.keras.optimizers.AdamW(learning_rate=new_lr, weight_decay=1e-5)

                # Update state
                self.current_optimizer = new_optimizer
                self.current_lr = new_lr
                self.wait = 0

                if self.verbose:
                    print(f"Changes applied. Now using {self.current_optimizer} with lr={self.current_lr}")


def build_model(hyperparams):
    """
    Build the model using the best hyperparameters
    """
    model = create_transformer(
        input_shape=(7200,),
        d_model=hyperparams['d_model'],
        num_heads=hyperparams['num_heads'],
        dff=hyperparams['dff'],
        num_layers=hyperparams['num_layers'],
        target_shape=(144, 1),
        rate=hyperparams['dropout_rate']
    )

    return model


def train_model_with_advanced_strategy(model, train_dataset, val_dataset, steps_per_epoch=None):
    """
    Train model with advanced optimizer and learning rate cycling strategy
    """
    # Initial learning rate
    initial_lr = 0.001

    # Set up initial optimizer (adam)
    primary_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    # Prepare callbacks
    callbacks = [
        # Advanced training callback that handles optimizer and learning rate cycling
        AdvancedTrainingCallback(
            patience=CONFIG['training']['optimizer_patience'],
            verbose=1
        ),

        # Early stopping with high patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['training']['early_stop_patience'],
            restore_best_weights=True,
            verbose=1
        ),

        # Save model checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            f"{CONFIG['training']['save_path']}_checkpoint.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=f"./{CONFIG['paths']['logs_dir']}/{CONFIG['training']['save_path']}",
            histogram_freq=1
        )
    ]

    # Compile model with primary optimizer
    model.compile(
        optimizer=primary_optimizer,
        loss=tf.keras.losses.Huber(),  # Simple Huber loss
        metrics=[
            'mae',  # Normalized MAE
            'mse',  # Normalized MSE
            real_speed_mae,  # Real wind speed MAE
            real_speed_mse,  # Real wind speed MSE
            power_mae,  # Power (V³) MAE
            power_rmse,  # Power (V³) RMSE
        ]
    )

    # Set up training parameters
    fit_kwargs = {
        'epochs': CONFIG['training']['epochs'],
        'callbacks': callbacks,
        'verbose': 1
    }

    # Add steps_per_epoch if provided
    if steps_per_epoch is not None:
        fit_kwargs.update({
            'steps_per_epoch': steps_per_epoch['train'],
            'validation_steps': steps_per_epoch['val']
        })
        print(f"Using steps_per_epoch: train={steps_per_epoch['train']}, val={steps_per_epoch['val']}")

    # Train the model
    print("\n=== Starting training with advanced optimizer and learning rate cycling ===")
    print(f"Initial optimizer: adam")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Strategy: After {CONFIG['training']['optimizer_patience']} epochs without improvement:")
    print(f"  - First, switch to alternative optimizer (adam ↔ adamw)")
    print(f"  - Then, if still no improvement, double learning rate")
    print(f"Early stopping patience: {CONFIG['training']['early_stop_patience']} epochs")

    # Clear GPU memory before training
    tf.keras.backend.clear_session()

    # Start training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        **fit_kwargs
    )

    return model, history


def plot_training_progress(history, fold_idx=None):
    """Enhanced plotting function with all metrics"""
    metrics_to_plot = [
        ('loss', 'val_loss', 'Loss'),
        ('mae', 'val_mae', 'Normalized MAE'),
        ('real_speed_mae', 'val_real_speed_mae', 'Real Speed MAE'),
        ('power_mae', 'val_power_mae', 'Power MAE'),
        ('power_rmse', 'val_power_rmse', 'Power RMSE')
    ]

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)))

    if len(metrics_to_plot) == 1:
        axes = [axes]  # Make it iterable if only one metric

    for i, (train_metric, val_metric, title) in enumerate(metrics_to_plot):
        if train_metric in history.history and val_metric in history.history:
            axes[i].plot(history.history[train_metric], label='Train')
            axes[i].plot(history.history[val_metric], label='Validation')
            axes[i].set_title(title)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(title)
            axes[i].legend()
            axes[i].grid(True)

    plt.tight_layout()

    # Save with fold info if provided
    if fold_idx is not None:
        plt.savefig(f"training_progress_fold{fold_idx}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig("training_progress.png", dpi=300, bbox_inches='tight')

    plt.close()


def plot_forecast(real_values, forecasted_values, time_range, fold_idx=None):
    """
    Plots real vs forecasted wind speeds.
    """
    if len(real_values) != len(time_range) or len(forecasted_values) != len(time_range):
        raise ValueError("Length mismatch between values and time range")

    denormalized_real = denorm_tf(real_values)
    denormalized_forecast = denorm_tf(forecasted_values)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_range, denormalized_real, 'b--', label='Actual Wind Speed', linewidth=2)
    ax.plot(time_range, denormalized_forecast, 'r-', label='Forecasted Wind Speed', linewidth=2)

    ax.set_title('Wind Speed Forecast vs Actual', fontsize=14, pad=20)
    ax.set_xlabel('Time (10-minute intervals)', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    y_min = min(np.min(denormalized_real), np.min(denormalized_forecast))
    y_max = max(np.max(denormalized_real), np.max(denormalized_forecast))
    padding = (y_max - y_min) * 0.1
    ax.set_ylim([y_min - padding, y_max + padding])

    plt.tight_layout()

    # Save with fold info if provided
    if fold_idx is not None:
        plt.savefig(f"wind_speed_forecast_fold{fold_idx}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig("wind_speed_forecast.png", dpi=300, bbox_inches='tight')

    return fig, ax


def evaluate_model(model, test_dataset):
    """Enhanced evaluation with comprehensive metrics"""
    results = model.evaluate(test_dataset, return_dict=True)

    metrics_mapping = {
        'loss': 'loss',  # Loss function
        'normalized_mae': 'mae',  # Original normalized MAE
        'normalized_mse': 'mse',  # Original normalized MSE
        'real_speed_mae': 'real_speed_mae',  # Denormalized wind speed MAE
        'real_speed_mse': 'real_speed_mse',  # Denormalized wind speed MSE
        'power_mae': 'power_mae',  # Power (V³) MAE
        'power_rmse': 'power_rmse'  # Power (V³) RMSE
    }

    metrics = {}
    for our_name, metric_name in metrics_mapping.items():
        if metric_name in results:
            metrics[our_name] = results[metric_name]

    print("\nModel Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    return metrics


def create_fold_datasets_from_test(test_data, num_folds=3):
    """
    Create fold datasets from test data for demonstration purposes
    when the fold_datasets.pkl file is missing.
    """
    print("⚠️ fold_datasets.pkl not found or corrupted. Creating fold datasets from test data...")

    X_test = test_data['X_test']
    y_test = test_data['y_test']

    # Get total size
    total_size = len(X_test)
    fold_size = total_size // num_folds

    fold_datasets = []
    steps_per_epoch = {}

    # Create folds
    for i in range(num_folds):
        # Define start and end indices for validation data
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < num_folds - 1 else total_size

        # Split data into train and validation
        val_indices = list(range(val_start, val_end))
        train_indices = [j for j in range(total_size) if j not in val_indices]

        X_train = X_test[train_indices]
        y_train = y_test[train_indices]
        X_val = X_test[val_indices]
        y_val = y_test[val_indices]

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .batch(CONFIG['training']['batch_size']) \
            .prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
            .batch(CONFIG['training']['batch_size']) \
            .prefetch(tf.data.AUTOTUNE)

        fold_datasets.append((train_dataset, val_dataset))

    # Calculate steps per epoch
    steps_per_epoch = {
        'train': (total_size - fold_size) // CONFIG['training']['batch_size'],
        'val': fold_size // CONFIG['training']['batch_size']
    }

    # Create dataset_info
    dataset_info = {
        'steps_per_epoch': steps_per_epoch,
        'num_folds': num_folds,
        'total_size': total_size,
        'fold_size': fold_size
    }

    print(f"Created {num_folds} fold datasets from test data")
    print(f"Train steps per epoch: {steps_per_epoch['train']}")
    print(f"Validation steps per epoch: {steps_per_epoch['val']}")

    return fold_datasets, dataset_info


def load_fold_datasets(fold_datasets_path):
    """
    Load fold datasets from a dictionary format
    """
    try:
        # Check file exists and load
        if not os.path.exists(fold_datasets_path):
            raise FileNotFoundError(f"Fold datasets file not found: {fold_datasets_path}")

        with open(fold_datasets_path, "rb") as f:
            data = pickle.load(f)

        # Check it's a dictionary with expected keys
        if not isinstance(data, dict):
            raise ValueError("Fold datasets should be a dictionary")

        required_keys = {'X_train', 'y_train', 'dataset_info'}
        if not all(key in data for key in required_keys):
            raise ValueError(f"Missing required keys. Found: {set(data.keys())}")

        # Create fold datasets
        X_train = data['X_train']
        y_train = data['y_train']
        dataset_info = data['dataset_info']

        # Convert to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .batch(CONFIG['training']['batch_size']) \
            .prefetch(tf.data.AUTOTUNE)

        # Creating a list with a single fold dataset
        fold_datasets = [(train_dataset, train_dataset)]  # Using train_dataset for both train and val

        return fold_datasets, dataset_info

    except Exception as e:
        print(f"❌ Error loading fold datasets: {type(e).__name__}")
        print(f"Detailed error: {str(e)}")
        raise

def main():
    try:
        # Create directories
        os.makedirs(CONFIG['paths']['logs_dir'], exist_ok=True)

        # Load test dataset
        print("\nLoading test dataset...")
        with open(CONFIG['data']['test_data_path'], "rb") as f:
            test_data = pickle.load(f)
            X_test = test_data['X_test']
            y_test = test_data['y_test']

        # Create test dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
            .batch(CONFIG['training']['batch_size']) \
            .prefetch(tf.data.AUTOTUNE)

        # Load fold datasets with comprehensive error checking
        try:
            fold_datasets, dataset_info = load_fold_datasets(CONFIG['data']['fold_datasets_path'])
        except Exception as load_error:
            print("Cannot proceed due to fold datasets loading error.")

        # Create a dictionary of hyperparameters
        hyperparams = {
            'd_model': CONFIG['model']['d_model'],
            'num_heads': CONFIG['model']['num_heads'],
            'dff': CONFIG['model']['dff'],
            'num_layers': CONFIG['model']['num_layers'],
            'dropout_rate': CONFIG['model']['dropout_rate']
        }

        print("\n=== Model Configuration ===")
        for param, value in hyperparams.items():
            print(f"{param}: {value}")

        # Train on each fold
        fold_models = []
        for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
            print(f"\n\n======== Training on Fold {fold_idx + 1} ========")

            # Build a fresh model for each fold
            model = build_model(hyperparams)

            # Train with advanced strategy
            trained_model, history = train_model_with_advanced_strategy(
                model,
                train_dataset,
                val_dataset,
                steps_per_epoch=dataset_info['steps_per_epoch']
            )

            # Plot training progress for each fold
            plot_training_progress(history, fold_idx=fold_idx + 1)

            # Save fold model
            save_path = f"{CONFIG['training']['save_path']}_fold{fold_idx + 1}.keras"
            trained_model.save(save_path)
            print(f"✅ Fold {fold_idx + 1} model saved to {save_path}")

            # Evaluate fold model
            print(f"\nEvaluating fold {fold_idx + 1} model on test dataset...")
            fold_metrics = evaluate_model(trained_model, test_dataset)

            # Create fold forecast visualization
            for batch_data in test_dataset.take(1):
                input_data, actual_values = batch_data
                forecasted_values = trained_model.predict(input_data)

                # Plot first sample from batch
                time_range = range(len(actual_values[0]))
                plot_forecast(
                    actual_values[0],
                    forecasted_values[0],
                    time_range,
                    fold_idx=fold_idx + 1
                )
                print(f"✅ Fold {fold_idx + 1} forecast plot saved")

            # Keep track of models
            fold_models.append((trained_model, fold_metrics))

            # Clean up GPU memory between folds
            tf.keras.backend.clear_session()

        # Select the best fold model based on validation metrics
        best_fold_idx = 0
        best_metric = float('inf')
        for i, (_, metrics) in enumerate(fold_models):
            # Use loss as the selection criterion
            if metrics['loss'] < best_metric:
                best_metric = metrics['loss']
                best_fold_idx = i

        print(f"\n=== Best model is from fold {best_fold_idx + 1} with loss: {best_metric:.6f} ===")

        # Save the best model as the final model
        best_model = fold_models[best_fold_idx][0]
        best_model.save(CONFIG['paths']['model_path'])
        print(f"✅ Best model (fold {best_fold_idx + 1}) saved as final model")

        print("\n=== Training completed successfully! ===")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
