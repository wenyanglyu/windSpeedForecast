# Nelson Wind Speed Forecast Project

This project is to forecast the wind speed in 24 hours 10-min interval and wind power(G128/4500 Gamesa turbine) as shown below:  
![day_4263_comparison](https://github.com/user-attachments/assets/ca69f655-5734-48d6-aacb-bb1659e5f58a)

## Environment
- Python 3.12.9
- GPU: RTX 4080
- Memory: 16GB
- OS: Linux
- Packages: See requirements.txt

## Data Sources
- Raw data hosted on Google Drive
- Automatically downloaded when preprocessing script runs


## Project Files

### 1. data_preprocessing.py
- Normalizes data
- Creates sliding windows
- Generates datasets for training

Run:
```bash
python data_preprocessing.py
```

Outputs:
- `fold_datasets.pkl`: For training
- `test_dataset.pkl`: For testing

Configuration:
- `batch_size`: Smaller = less GPU memory but slower processing, larger = more accurate result but the GPU memory may be not enough, current design 64 is the best for GTX4080
- `window_shift`: Smaller = more data, higher similarity; current design is 3

### 2. model_builder.py
This file defines the Transformer model architecture. No modifications needed.

Functions:
- Transformer layers implementation
- Model creation, saving and loading

### 3. optuna_trial.py
Hyperparameter optimization using Optuna.

- Run this file 5 times sequentially (20 epochs per run)
- Prevents OOM by splitting 100 epochs into smaller batches

Run:
```bash
python optuna_trial.py
```

### 4. optuna_analysis.py
Analyzes the hyperparameter search results.

Run:
```bash
python optuna_analysis.py
```

### 5. training.py
Formal training with best hyperparameters.

Configuration:
- `epochs`: 2000
- `optimizer_patience`: 10 - Switch optimizer after 10 epochs without improvement
- `early_stop_patience`: 25 - Stop training if no improvement for 25 epochs
- `batch_size`: 64, Bigger size have better performance but also require higher GPU memory
  
Hyperparameters:
```python
'model': {
    # Best hyperparameters from analysis
    'd_model': 320,  # Using top performer from group analysis
    'num_heads': 16,  # Using top performer from group analysis
    'dff': 768,  # Using top performer from group analysis
    'num_layers': 2,  # Using top performer from group analysis
    'dropout_rate': 0.0521  # From best trial
}
```
Manually modify these values if needed.

Run:
```bash
python training.py
```

### 6. forecast.py
Evaluates model performance and generates forecasts.

Features:
- Loads the trained model
- Calculates wind power from wind speed predictions
- Generates comparison plots for actual vs predicted values
- Creates detailed metrics reports

Run:
```bash
python forecast.py
```

Outputs:
- Wind speed and power comparison plots
- Detailed metrics in forecast_metrics.txt

### 7. data_analysis folder
Evaluates features relationship, correlation and importance.
