import numpy as np
import matplotlib.pyplot as plt

# Generate values for a full year (365 days)
days = np.arange(1, 366)
day_sin = np.sin(2 * np.pi * days / 365)
day_cos = np.cos(2 * np.pi * days / 365)

# Generate values for a full day (144 ten-minute intervals)
minutes = np.arange(0, 1440, 10)
time_sin = np.sin(2 * np.pi * minutes / 1440)
time_cos = np.cos(2 * np.pi * minutes / 1440)

# Plot day encoding
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(days, day_sin, label="sin(day)")
plt.plot(days, day_cos, label="cos(day)")
plt.xlabel("Day of Year")
plt.ylabel("Encoded Value")
plt.title("Cyclic Encoding for Day of the Year")
plt.legend()
plt.grid()

# Plot time encoding
plt.subplot(1, 2, 2)
plt.plot(minutes, time_sin, label="sin(time)")
plt.plot(minutes, time_cos, label="cos(time)")
plt.xlabel("Minute of Day")
plt.ylabel("Encoded Value")
plt.title("Cyclic Encoding for Time of the Day")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
