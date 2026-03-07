#!/usr/bin/env python3
"""Test 5-convolve.py with multiple kernels"""

import numpy as np
import matplotlib.pyplot as plt
convolve = __import__('5-convolve').convolve

# Random test data
np.random.seed(0)
m, h, w, c = 2, 32, 32, 3       # 2 images, 32x32, 3 channels
nc = 3                            # 3 kernels
kh, kw = 3, 3

images = np.random.randint(0, 256, (m, h, w, c))
kernels = np.random.randint(-1, 2, (kh, kw, c, nc))

conv_ims = convolve(images, kernels, padding='valid')

print("Input images shape:", images.shape)
print("Kernels shape:", kernels.shape)
print("Convolved output shape:", conv_ims.shape)

plt.imshow(images[0])
plt.show()

# Show output for each kernel of first image
for i in range(nc):
    plt.imshow(conv_ims[0, :, :, i])
    plt.title("Kernel {}".format(i))
    plt.show()
