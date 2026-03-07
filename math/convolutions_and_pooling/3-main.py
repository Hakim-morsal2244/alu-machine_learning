#!/usr/bin/env python3
"""Sandbox test for strided convolution on small grayscale images."""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

# ===== Dynamically load the module =====
module_name = "convolve3"
file_path = "./3-convolve_grayscale.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
convolve_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = convolve_module
spec.loader.exec_module(convolve_module)

convolve_grayscale = convolve_module.convolve_grayscale

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

# Stride and padding
stride = (2, 2)
padding = 'valid'

# Perform strided convolution
images_conv = convolve_grayscale(images, kernel, padding=padding, stride=stride)

print("Input images shape:", images.shape)
print("Kernel shape:", kernel.shape)
print("Stride:", stride)
print("Padding:", padding)
print("Convolved output shape:", images_conv.shape)

# Show first image and its convolution
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(images[0], cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Strided Convolution")
plt.imshow(images_conv[0], cmap='gray')

plt.show()
