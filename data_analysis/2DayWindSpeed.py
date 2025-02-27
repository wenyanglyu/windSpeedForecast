import pandas as pd
import matplotlib.pyplot as plt

# Function to load wind speed data from a local file
def load_wind_data(file_path):
    """
    Loads wind speed data from a local CSV file.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['OBS_DATE'], errors='coerce')
    df.set_index('timestamp', inplace=True)
    df['MEAN_SPD10'] = pd.to_numeric(df['MEAN_SPD10'], errors='coerce')
    df.dropna(subset=['MEAN_SPD10'], inplace=True)
    return df

# Load and filter the dataset for 2022-2023
file_path = r"/home/em19255/Downloads/40751_Nelson_allFeatures.csv"  # Use your local file
df = load_wind_data(file_path)

# Function to plot 2-day average wind speed
def plot_two_day_avg_wind_speed(df):
    """
    Plots the 2-day average wind speed from Jan 2022 to Dec 2023,
    with the x-axis labeled by months.
    """
    # Ensure datetime format
    df['OBS_DATE'] = pd.to_datetime(df.index, errors='coerce')

    # Filter data for 2022-2023
    df_filtered = df[(df.index >= '2022-01-01') & (df.index <= '2023-12-31')].copy()

    if df_filtered.empty:
        print("No data available for the selected time range.")
        return

    # Compute 2-day average wind speed
    df_two_day_avg = df_filtered.resample('2D').mean()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df_two_day_avg.index, df_two_day_avg['MEAN_SPD10'], color='blue', alpha=0.7, linewidth=0.8)

    # X-Axis Formatting: Show only month labels
    plt.xticks(pd.date_range(start="2022-01-01", end="2024-01-01", freq="M"),
               labels=pd.date_range(start="2022-01-01", end="2024-01-01", freq="M").strftime("%b %Y"),
               rotation=45)

    # Labels and title
    plt.xlabel("Month (2022-2023)")
    plt.ylabel("2-Day Average Wind Speed (m/s)")
    plt.title("2-Day Average Wind Speed from 2022 to 2023")
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

# Example usage:
plot_two_day_avg_wind_speed(df)
