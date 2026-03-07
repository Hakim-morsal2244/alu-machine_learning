#!/usr/bin/env python3
"""Test 6-pool.py"""

import numpy as np
import matplotlib.pyplot as plt
pool = __import__('6-pool').pool

# Random test data
np.random.seed(0)
m, h, w, c = 2, 32, 32, 3       # 2 images, 32x32, 3 channels
images = np.random.randint(0, 256, (m, h, w, c))

kernel_shape = (2, 2)
stride = (2, 2)

# Max pooling
pooled_max = pool(images, kernel_shape, stride, mode='max')
# Average pooling
pooled_avg = pool(images, kernel_shape, stride, mode='avg')

print("Input images shape:", images.shape)
print("Max pooled shape:", pooled_max.shape)
print("Avg pooled shape:", pooled_avg.shape)

plt.imshow(images[0])
plt.title("Original Image")
plt.show()

plt.imshow(pooled_max[0] / 255)
plt.title("Max Pooled")
plt.show()

plt.imshow(pooled_avg[0] / 255)
plt.title("Avg Pooled")
plt.show()
