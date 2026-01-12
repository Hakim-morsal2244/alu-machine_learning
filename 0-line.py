#!/usr/bin/env python3
"""
0-line.py
Plot a cubic line graph from 0 to 10 as a solid red line and save it as an image.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid display errors
import matplotlib.pyplot as plt

# Create y-values (cubes of 0 to 10)
y = np.arange(0, 11) ** 3

# Create x-values (0 to 10)
x = np.arange(0, 11)

# Plot y vs x as a solid red line
plt.plot(x, y, 'r')

# Set x-axis limits from 0 to 10
plt.xlim(0, 10)

# Save the graph as an image
plt.savefig("cubic_graph.png")

# Optional: display the plot if GUI is available
# plt.show()
