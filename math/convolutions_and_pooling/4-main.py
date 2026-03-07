#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels

# Test with random images
np.random.seed(0)
m, h, w, c = 10, 32, 32, 3  # 10 images, 32x32, 3 channels
images = np.random.randint(0, 256, (m, h, w, c))
kernel = np.array([
    [[0, 0, 0], [-1, -1, -1], [0, 0, 0]],
    [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]],
    [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]
])

images_conv = convolve_channels(images, kernel, padding='valid')
print(images_conv.shape)

plt.imshow(images[0].astype(np.uint8))
plt.show()
plt.imshow(images_conv[0], cmap='gray')
plt.show()
