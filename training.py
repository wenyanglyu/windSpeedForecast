import tensorflow as tf
import optuna
import numpy as np
import matplotlib.pyplot as plt

from model_builder import create_transformer, combined_loss, power_loss
from forecast import denormalize_wind_speed  # Import denormalization function


# Custom metrics for real wind speed and power
def denorm_tf(speed):
    """
    Denormalize wind speed using pure TensorFlow operations.

    Args:
        speed: Normalized wind speed tensor
    Returns:
        Denormalized wind speed tensor
    """
    speed = tf.cast(tf.squeeze(speed), tf.float32)

    speed_30 = speed * 30.0  # For speed < 0.1
    speed_normal = 3.0 + (speed - 0.1) * (15.0 - 3.0) / 0.8  # For 0.1 <= speed <= 0.9
    speed_high = 15.0 + (speed - 0.9) * (20.0 - 15.0) / 0.1  # For 0.9 < speed <= 1.0

    result = tf.where(speed < 0.1, speed_30,
                      tf.where(speed <= 0.9, speed_normal,
                               tf.where(speed <= 1.0, speed_high,
                                        20.0 * tf.ones_like(speed))))

    return result


def real_speed_mae(y_true, y_pred):
    """Calculate MAE using denormalized wind speeds"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    y_true_denorm = denorm_tf(y_true)
    y_pred_denorm = denorm_tf(y_pred)

    return tf.reduce_mean(tf.abs(y_true_denorm - y_pred_denorm))


def real_speed_mse(y_true, y_pred):
    """Calculate MSE using denormalized wind speeds"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    y_true_denorm = denorm_tf(y_true)
    y_pred_denorm = denorm_tf(y_pred)

    return tf.reduce_mean(tf.square(y_true_denorm - y_pred_denorm))


def power_mae(y_true, y_pred):
    """Calculate MAE of wind power (V³)"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    y_true_denorm = denorm_tf(y_true)
    y_pred_denorm = denorm_tf(y_pred)

    y_true_power = tf.pow(y_true_denorm, 3)
    y_pred_power = tf.pow(y_pred_denorm, 3)
    return tf.reduce_mean(tf.abs(y_true_power - y_pred_power))


def power_rmse(y_true, y_pred):
    """Calculate RMSE of wind power (V³)"""
    y_true = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

    y_true_denorm = denorm_tf(y_true)
    y_pred_denorm = denorm_tf(y_pred)

    y_true_power = tf.pow(y_true_denorm, 3)
    y_pred_power = tf.pow(y_pred_denorm, 3)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true_power - y_pred_power)))

# Modify objective function to include new metrics
def objective(trial, train_dataset, val_dataset, optuna_epochs, steps_per_epoch):
    """
    Modified objective function with proper error handling and logging
    """
    try:
        # Sample hyperparameters
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        dff = trial.suggest_categorical("dff", [128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)

        # Create model with proper input shape
        model = create_transformer(
            input_shape=(6039,),
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            num_layers=num_layers,
            target_shape=(144, 1),
            rate=dropout_rate
        )
        save_path = 'wind_forecast_model'

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"{save_path}_checkpoint.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/{save_path}',
                histogram_freq=1
            )
        ]

        # Train model with steps_per_epoch
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=optuna_epochs,
            steps_per_epoch=steps_per_epoch['train'],
            validation_steps=steps_per_epoch['val'],
            callbacks=callbacks,
            verbose=1
        )

        # Get best validation loss
        val_loss = min(history.history['val_loss'])

        # Save trial information
        trial.set_user_attr("best_val_loss", val_loss)
        trial.set_user_attr("best_model_config", model.get_config())

        return val_loss

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()


def run_optuna_optimization(train_dataset, val_dataset, optuna_epochs, n_trials, steps_per_epoch):
    study_name = "wind_forecast_optimization"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage="sqlite:///wind_forecast.db",  # Save study results
        load_if_exists=True
    )
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, optuna_epochs, steps_per_epoch),
        n_trials=n_trials
    )

    # Save best trial info
    print(f"Best trial value: {study.best_value}")
    print(f"Best trial params: {study.best_params}")

    return study.best_params
# Modified train_best_model function
def train_best_model(best_hyperparams, train_dataset, val_dataset, epochs=200, steps_per_epoch=None,
                     save_path="best_model"):
    """
    Enhanced training function with better monitoring and control

    Args:
        best_hyperparams: Dictionary of best hyperparameters from optuna
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        steps_per_epoch: Dictionary containing steps per epoch for training and validation
        save_path: Path to save the model
    """
    try:
        print("\nIn train_best_model:")
        print(f"Received steps_per_epoch: {steps_per_epoch}")

        best_model = create_transformer(
            input_shape=(6039,),
            d_model=best_hyperparams["d_model"],
            num_heads=best_hyperparams["num_heads"],
            dff=best_hyperparams["dff"],
            num_layers=best_hyperparams["num_layers"],
            target_shape=(144, 1),
            rate=best_hyperparams["dropout_rate"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"{save_path}_checkpoint.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/{save_path}',
                histogram_freq=1
            )
        ]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        best_model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[
                'mae',  # Normalized MAE
                'mse',  # Normalized MSE
                real_speed_mae,  # Real wind speed MAE
                real_speed_mse,  # Real wind speed MSE
                power_mae,  # Power (V³) MAE
                power_rmse,  # Power (V³) RMSE
            ]
        )

        # Add steps_per_epoch to model.fit if provided
        fit_kwargs = {
            'epochs': epochs,
            'callbacks': callbacks,
            'verbose': 1,
            'steps_per_epoch': steps_per_epoch['train'],
            'validation_steps': steps_per_epoch['val']
        }

        if steps_per_epoch is not None:
            fit_kwargs.update({
                'steps_per_epoch': steps_per_epoch['train'],
                'validation_steps': steps_per_epoch['val']
            })
            print(f"\nFit kwargs after adding steps:")
            print(f"steps_per_epoch: {fit_kwargs.get('steps_per_epoch')}")
            print(f"validation_steps: {fit_kwargs.get('validation_steps')}")
        else:
            print("\nWarning: steps_per_epoch is None!")

        print("\nFinal fit_kwargs:")
        print(fit_kwargs)

        history = best_model.fit(
            train_dataset,
            validation_data=val_dataset,
            **fit_kwargs
        )

        return best_model, history

    except Exception as e:
        print(f"Training failed: {str(e)}")
        tf.keras.backend.clear_session()
        raise


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

    print("\nAvailable metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    return metrics



def plot_training_progress(history):
    """Enhanced plotting function with all metrics"""
    metrics_to_plot = [
        ('loss', 'val_loss', 'Loss'),
        ('mae', 'val_mae', 'Normalized MAE'),
        ('real_speed_mae', 'val_real_speed_mae', 'Real Speed MAE'),
        ('power_mae', 'val_power_mae', 'Power MAE'),
        ('power_rmse', 'val_power_rmse', 'Power RMSE')
    ]

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)))

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
    plt.savefig("training_progress.png", dpi=300, bbox_inches='tight')
    plt.close()