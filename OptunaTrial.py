import optuna
import tensorflow as tf
from model_builder import create_transformer
import os
import optuna.visualization.matplotlib as optuna_vis
import pickle
import gc

CONFIG = {
    'data': {
        'test_data_path': "test_dataset.pkl",
        'optuna_results_path': "optuna_results.pkl",
        'batch_size': 32
    },
    'training': {
        'optuna_epochs': 20,  # Epochs per trial
        'n_trials_per_run': 20,  # Number of trials per run
        'total_runs': 5,  # 5 runs of 20 epochs = 100 epochs
        'save_dir': "optuna_results"  # Folder to store results (✅ ADDED)
    }
}


def get_gpu_memory_info():
    """Get GPU memory usage information."""
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        print("Available GPUs:", gpu_devices)

        if not gpu_devices:
            return "No GPU available"

        memory_info = []
        for idx, device in enumerate(gpu_devices):
            try:
                # Use the correct device name format
                memory = tf.config.experimental.get_memory_info(f'/device:GPU:{idx}')
                memory_info.append({
                    'device': f'GPU:{idx}',
                    'current': memory['current'] / (1024 ** 2),  # Convert to MB
                    'peak': memory['peak'] / (1024 ** 2),
                    'total': 16376  # Your GPU's total memory in MB
                })
            except Exception as e:
                print(f"Error getting memory for GPU:{idx}: {e}")

        return memory_info
    except Exception as e:
        return f"Error getting GPU memory info: {str(e)}"

def objective(trial, train_dataset, val_dataset, optuna_epochs, steps_per_epoch):
    """
    Objective function for Optuna optimization with memory management.
    """
    try:
        print("\nCleaning GPU memory before trial...")
        tf.keras.backend.clear_session()
        gc.collect()

        print("\nGPU Memory at trial start:")
        print(get_gpu_memory_info())

        # Suggest hyperparameters
        d_model = trial.suggest_int("d_model", 64, 512, step=64)
        num_heads = trial.suggest_int("num_heads", 2, 16, step=2)
        dff = trial.suggest_int("dff", 128, 1024, step=128)
        num_layers = trial.suggest_int("num_layers", 2, 6)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.25)
        loss_function = trial.suggest_categorical("loss_function", ["mae", "mse", "huber"])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop", "sgd"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

        # Define optimizer
        optimizer = {
            "adam": tf.keras.optimizers.Adam(learning_rate=learning_rate),
            "adamw": tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        }[optimizer_name]

        # Define loss function
        loss = {
            "mae": tf.keras.losses.MeanAbsoluteError(),
            "mse": tf.keras.losses.MeanSquaredError(),
            "huber": tf.keras.losses.Huber(delta=1.0),
        }[loss_function]

        # Create and compile model
        model = create_transformer(
            input_shape=(7200,),
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            num_layers=num_layers,
            target_shape=(144, 1),
            rate=dropout_rate
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
        ]

        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=optuna_epochs,
            steps_per_epoch=steps_per_epoch['train'],
            validation_steps=steps_per_epoch['val'],
            callbacks=callbacks,
            verbose=1
        )

        # ✅ Capture Validation Loss
        val_loss = min(history.history["val_loss"])

        # ✅ Store hyperparameters + val_loss
        trial_results = {
            "d_model": d_model,
            "num_heads": num_heads,
            "dff": dff,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "loss_function": loss_function,
            "val_loss": val_loss  #
        }

        # Save results to pkl file
        save_optuna_trial(trial_results)

        print("\nCleaning GPU memory after trial...")
        tf.keras.backend.clear_session()
        gc.collect()

        return val_loss

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()


def save_optuna_trial(trial_results):
    """Append Optuna trial results to optuna_results.pkl without overwriting previous trials"""
    file_path = os.path.join(CONFIG['training']['save_dir'], "optuna_results.pkl")
    os.makedirs(CONFIG['training']['save_dir'], exist_ok=True)

    # Initialize list for all trials
    all_trials = []

    # Load existing trials if file exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            with open(file_path, "rb") as f:
                all_trials = pickle.load(f)  # Load the entire list at once

            if not isinstance(all_trials, list):
                # If it's not a list (might be a single trial), convert to list
                all_trials = [all_trials]
        except Exception as e:
            print(f"Error loading existing trials: {e}")
            all_trials = []

    # Append new trial
    all_trials.append(trial_results)

    # Save all trials as a single list object
    with open(file_path, "wb") as f:
        pickle.dump(all_trials, f)

    print(f"✅ Saved trial {len(all_trials)} to {file_path}")
    print(f"Total trials in file: {len(all_trials)}")




def get_next_file():
    """Always return the same file: optuna_results.pkl"""
    os.makedirs(CONFIG['training']['save_dir'], exist_ok=True)
    return os.path.join(CONFIG['training']['save_dir'], "optuna_results.pkl")




def load_optuna_study():
    """Load an existing Optuna study or create a new one."""
    study_name = "wind_forecast_optimization"
    storage_path = "sqlite:///wind_forecast.db"

    try:
        if os.path.exists("wind_forecast.db"):
            print("Found existing Optuna study. Resuming...")
            return optuna.load_study(study_name=study_name, storage=storage_path)
        else:
            print("No existing study found. Starting fresh...")
            return optuna.create_study(study_name=study_name, direction="minimize", storage=storage_path)
    except Exception as e:
        print(f"Error loading Optuna study: {e}. Restarting study.")
        return optuna.create_study(study_name=study_name, direction="minimize", storage=storage_path)


def run_optuna_incremental(train_dataset, val_dataset, steps_per_epoch):
    """Run Optuna tuning incrementally based on existing files."""
    file_path = get_next_file()
    if file_path is None:
        print("All Optuna runs are already completed. Exiting.")
        return

    print(f"Running Optuna Hyperparameter Search: {file_path}")

    study = load_optuna_study()

    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, CONFIG['training']['optuna_epochs'],
                                steps_per_epoch),
        n_trials=CONFIG['training']['n_trials_per_run'],
        callbacks=[lambda study, trial: print(f"\nCompleted Trial {trial.number}")],
        show_progress_bar=True
    )

    # ✅ Prevent crash if no valid trials exist
    if len(study.trials) == 0 or study.best_trial is None:
        print("No valid trials found. Skipping saving best parameters.")
        return

def get_completed_runs():
    """Read the last completed run from a tracking file."""
    progress_file = os.path.join(CONFIG['training']['save_dir'], "optuna_progress.txt")

    if not os.path.exists(progress_file):
        return 0  # No runs completed yet

    with open(progress_file, "r") as f:
        try:
            return int(f.read().strip())  # Read and convert to integer
        except ValueError:
            return 0  # If file is corrupted, start from 0

def update_completed_runs(run_number):
    """Update the tracking file with the last completed run."""
    progress_file = os.path.join(CONFIG['training']['save_dir'], "optuna_progress.txt")
    with open(progress_file, "w") as f:
        f.write(str(run_number))  # Save progress


def main():
    try:
        with open(CONFIG['data']['test_data_path'], "rb") as f:
            test_data = pickle.load(f)
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            dataset_info = test_data['dataset_info']

        train_size = int(0.8 * len(X_test))

        optuna_train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_test[:train_size], y_test[:train_size])
        ).batch(CONFIG['data']['batch_size']).prefetch(tf.data.AUTOTUNE)

        optuna_val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_test[train_size:], y_test[train_size:])
        ).batch(CONFIG['data']['batch_size']).prefetch(tf.data.AUTOTUNE)

        optuna_steps = {'train': train_size // 32, 'val': (len(X_test) - train_size) // 32}

        completed_runs = get_completed_runs()

        if completed_runs >= CONFIG['training']['total_runs']:
            print("All Optuna runs are completed. Exiting.")
            return

        print(f"\n========== Running Optuna Batch {completed_runs + 1} / {CONFIG['training']['total_runs']} ==========")

        run_optuna_incremental(optuna_train_dataset, optuna_val_dataset, optuna_steps)

        update_completed_runs(completed_runs + 1)

    except Exception as e:
        print(f"Optuna optimization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
