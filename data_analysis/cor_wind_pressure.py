import requests
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_csv(file_id):
    """
    Downloads the CSV from Google Drive and loads it into a DataFrame.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    csv_data = StringIO(response.text)
    data = pd.read_csv(csv_data, dtype=str)  # Read as string to detect non-numeric values
    return data

def preprocess_nelson_data(wind_data):
    """
    Preprocesses the Nelson wind and pressure data.
    """
    wind_data.columns = wind_data.columns.str.strip()
    wind_data['OBS_DATE'] = pd.to_datetime(wind_data['OBS_DATE'], errors='coerce')
    wind_data['MEAN_SPD10'] = pd.to_numeric(wind_data['MEAN_SPD10'], errors='coerce')
    wind_data['PRESSURE_MSL10'] = pd.to_numeric(wind_data['PRESSURE_MSL10'], errors='coerce')
    return wind_data

def preprocess_greymouth_data(pressure_data):
    """
    Preprocesses the Greymouth pressure data.
    """
    pressure_data.columns = pressure_data.columns.str.strip()
    pressure_data['OBS_DATE'] = pd.to_datetime(pressure_data['OBS_DATE'], errors='coerce')
    pressure_data['PRESSURE_MSL60'] = pd.to_numeric(pressure_data['PRESSURE_MSL60'], errors='coerce')
    return pressure_data

# Load and preprocess Nelson data
nelson_file_id = "1FftdSBsoQHrZ0dxnoYDAVYME3Gp8UXsb"
nelson_data = load_csv(nelson_file_id)
nelson_data_processed = preprocess_nelson_data(nelson_data)

# Load and preprocess Greymouth data
greymouth_file_id = "1WRW60JP8xOl0fcQIivJaUF8883k7Ii6e"
greymouth_data = load_csv(greymouth_file_id)
greymouth_data_processed = preprocess_greymouth_data(greymouth_data)

# Resample Greymouth data to 1-hour intervals
greymouth_data_processed.set_index('OBS_DATE', inplace=True)
greymouth_resampled = greymouth_data_processed.resample('h').mean(numeric_only=True)

# Expand Greymouth hourly pressure into 6x10-minute intervals
expanded_greymouth = greymouth_resampled.reindex(
    greymouth_resampled.resample('10min').ffill().index
).reset_index()

# Merge expanded Greymouth pressure with original Nelson 10-min data
merged_data = pd.merge_asof(
    nelson_data_processed.sort_values('OBS_DATE'),
    expanded_greymouth.sort_values('OBS_DATE'),
    on='OBS_DATE',
    direction='nearest'
)

# Forward fill missing values
merged_data = merged_data.ffill()

# Calculate the pressure difference
merged_data['pressure_diff'] = merged_data['PRESSURE_MSL10'] - merged_data['PRESSURE_MSL60']

# Filter data for January 2017
merged_data['year'] = merged_data['OBS_DATE'].dt.year
merged_data['month'] = merged_data['OBS_DATE'].dt.month
merged_data_filtered = merged_data[(merged_data['year'] == 2017) & (merged_data['month'] == 1)]

# Plotting
plt.figure(figsize=(15, 6))

# Plot wind speed in blue
plt.plot(merged_data_filtered['OBS_DATE'], merged_data_filtered['MEAN_SPD10'], color='blue', label='Wind Speed (m/s)')

# Plot pressure difference in red
plt.plot(merged_data_filtered['OBS_DATE'], merged_data_filtered['pressure_diff'], color='red', label='Pressure Difference (hPa)')

# Labeling the axes and the title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Wind Speed and Pressure Difference (Jan 2017)')
plt.legend()

# Display the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot with Regression Line
plt.figure(figsize=(10, 6))
sns.regplot(x=merged_data_filtered['pressure_diff'], y=merged_data_filtered['MEAN_SPD10'],
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.xlabel("Pressure Difference (hPa)")
plt.ylabel("Wind Speed (m/s)")
plt.title("Scatter Plot: Wind Speed vs. Pressure Difference (Jan 2017)")
plt.grid(True)
plt.show()

# Heatmap (Correlation Matrix)
plt.figure(figsize=(8, 6))
correlation_matrix = merged_data_filtered[['MEAN_SPD10', 'pressure_diff']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Jan 2017)")
plt.show()


# Convert pressure difference to absolute values
merged_data_filtered['abs_pressure_diff'] = np.abs(merged_data_filtered['pressure_diff'])

# Scatter Plot with Regression Line using Absolute Pressure Difference
plt.figure(figsize=(10, 6))
sns.regplot(x=merged_data_filtered['abs_pressure_diff'], y=merged_data_filtered['MEAN_SPD10'],
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.xlabel("Absolute Pressure Difference (hPa)")
plt.ylabel("Wind Speed (m/s)")
plt.title("Scatter Plot: Wind Speed vs. Absolute Pressure Difference (Jan 2017)")
plt.grid(True)
plt.show()

# Ensure absolute pressure difference is in the dataset
merged_data_filtered['abs_pressure_diff'] = np.abs(merged_data_filtered['pressure_diff'])

# Heatmap (Correlation Matrix with Absolute Pressure Difference)
plt.figure(figsize=(8, 6))
correlation_matrix = merged_data_filtered[['MEAN_SPD10', 'abs_pressure_diff']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Jan 2017) with Absolute Pressure Difference")
plt.show()
