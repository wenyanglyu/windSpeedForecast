import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Constants
CUT_IN_SPEED = 3.0  # m/s
RATED_SPEED = 15.0  # m/s
CUT_OUT_SPEED = 20.0  # m/s


def denormalize_wind_speed(normalized_wind_speed_series, min_speed=None, max_speed=None):
    """
    Denormalizes the wind speed based on the custom normalization logic.
    """
    # Convert tensor to numpy if it's a tensor
    if tf.is_tensor(normalized_wind_speed_series):
        normalized_wind_speed_series = normalized_wind_speed_series.numpy()

    # Use np.min/max instead of pandas min/max
    if min_speed is None or max_speed is None:
        min_speed = np.min(normalized_wind_speed_series)
        max_speed = np.max(normalized_wind_speed_series)

    # Denormalization logic remains the same
    denormalized_speed = np.where(
        np.isnan(normalized_wind_speed_series),
        np.nan,
        np.where(
            normalized_wind_speed_series < 0.1,
            normalized_wind_speed_series * CUT_IN_SPEED / 0.1,
            np.where(
                (normalized_wind_speed_series >= 0.1) & (normalized_wind_speed_series <= 0.9),
                CUT_IN_SPEED + (normalized_wind_speed_series - 0.1) * (RATED_SPEED - CUT_IN_SPEED) / 0.8,
                np.where(
                    (normalized_wind_speed_series > 0.9) & (normalized_wind_speed_series <= 1.0),
                    RATED_SPEED + (normalized_wind_speed_series - 0.9) * (CUT_OUT_SPEED - RATED_SPEED) / 0.1,
                    CUT_OUT_SPEED + (normalized_wind_speed_series - 1.0) * (max_speed - CUT_OUT_SPEED) / 0.1
                )
            )
        )
    )
    return denormalized_speed


def plot_forecast(real_values, forecasted_values, time_range):
    """
    Plots real vs forecasted wind speeds with proper denormalization.
    """
    if len(real_values) != len(time_range) or len(forecasted_values) != len(time_range):
        raise ValueError("Length mismatch between values and time range")

    denormalized_real = denormalize_wind_speed(real_values)
    denormalized_forecast = denormalize_wind_speed(forecasted_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_range, denormalized_real, 'b--', label='Real Wind Speed')
    ax.plot(time_range, denormalized_forecast, 'r-', label='Forecasted Wind Speed')

    ax.set_title('Wind Speed Forecast vs Actual')
    ax.set_xlabel('Time (10-minute intervals)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("wind_speed_forecast.png", dpi=300)
    return fig, ax


