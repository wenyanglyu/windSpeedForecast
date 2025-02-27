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

# Load and filter the dataset for 2023-Jan-01 to 2023-Jan-02
file_path = r"/home/em19255/Downloads/40751_Nelson_allFeatures.csv"  # Update your actual path
df = load_wind_data(file_path)

# Filter for the required two days
df_filtered = df[(df.index >= "2023-01-01") & (df.index < "2023-01-03")].copy()

# Ensure only numeric columns are resampled
df_resampled = df_filtered.resample('3H').mean(numeric_only=True)  # Resample every 3 hours

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['MEAN_SPD10'], color='blue', alpha=0.6, linewidth=0.5, label="10-Min Wind Speed")

# X-Axis Formatting: Show 3-hour intervals
plt.xticks(df_resampled.index, df_resampled.index.strftime('%H:%M'), rotation=45)

# Labels and title
plt.xlabel("Time (3-Hour Intervals)")
plt.ylabel("Wind Speed (m/s)")
plt.title("10-Min Interval Wind Speed (Jan 1 - Jan 2, 2023)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
