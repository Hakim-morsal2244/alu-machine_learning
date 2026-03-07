#!/usr/bin/env python3
"""Test valid convolution on small grayscale images (sandbox) with original file name."""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

# ===== Dynamically load the module =====
module_name = "convolve0"  # temporary name
file_path = "./0-convolve_grayscale_valid.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
convolve_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = convolve_module
spec.loader.exec_module(convolve_module)

convolve_grayscale_valid = convolve_module.convolve_grayscale_valid

# ===== Sandbox dataset =====
images = np.array([
    [[1, 2, 3, 0, 1],
     [0, 1, 2, 3, 1],
     [1, 0, 1, 2, 2],
     [2, 1, 0, 1, 0],
     [1, 2, 3, 1, 0]],

    [[0, 1, 0, 2, 1],
     [1, 2, 1, 0, 0],
     [0, 1, 2, 1, 1],
     [1, 0, 1, 2, 0],
     [2, 1, 0, 1, 1]]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Perform convolution
images_conv = convolve_grayscale_valid(images, kernel)

# Print shapes
print("Input images shape:", images.shape)
print("Kernel shape:", kernel.shape)
print("Convolved output shape:", images_conv.shape)

# Show first image and its convolution
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(images[0], cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Convolved")
plt.imshow(images_conv[0], cmap='gray')

plt.show()
