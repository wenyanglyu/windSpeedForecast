import numpy as np
import matplotlib.pyplot as plt

# Generate wind directions from 0 to 360 degrees
angles = np.linspace(0, 360, 100)

# Convert degrees to radians
radians = np.radians(angles)

# Compute sine and cosine encoding
sin_values = np.sin(radians)
cos_values = np.cos(radians)

# Plot the encoding
plt.figure(figsize=(10, 5))
plt.plot(angles, sin_values, label="sin(angle)", linestyle='-', marker='o')
plt.plot(angles, cos_values, label="cos(angle)", linestyle='-', marker='s')

# Highlight the problematic 179° and 181° transition
plt.axvline(179, color='gray', linestyle='--', alpha=0.5)
plt.axvline(181, color='gray', linestyle='--', alpha=0.5)
plt.scatter([179, 181], [np.sin(np.radians(179)), np.sin(np.radians(181))], color='red', label="sin(179°) & sin(181°)")
plt.scatter([179, 181], [np.cos(np.radians(179)), np.cos(np.radians(181))], color='blue', label="cos(179°) & cos(181°)")

# Labels and legends
plt.xlabel("Wind Direction (Degrees)")
plt.ylabel("Encoded Value")
plt.title("Wind Direction Encoding using Sine and Cosine")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
