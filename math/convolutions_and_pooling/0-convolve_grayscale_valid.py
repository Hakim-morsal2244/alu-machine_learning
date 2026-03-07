#!/usr/bin/env python3
"""Perform valid convolution on grayscale images using a kernel."""

import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Parameters
    ----------
    images : numpy.ndarray
        Numpy array of shape (m, h, w) containing m grayscale images
    kernel : numpy.ndarray
        Numpy array of shape (kh, kw) containing the kernel

    Returns
    -------
    numpy.ndarray
        Numpy array containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions for valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    # Perform convolution (only two loops allowed)
    for i in range(output_h):
        for j in range(output_w):
            # Slice all images at once
            image_region = images[:, i:i+kh, j:j+kw]
            # Element-wise multiply and sum over kh and kw for each image
            output[:, i, j] = np.sum(image_region * kernel, axis=(1, 2))

    return output

# ===== Sandbox Test (optional) =====
if __name__ == "__main__":
    # Small 2 images 5x5 for testing
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

    output = convolve_grayscale_valid(images, kernel)
    print("Input images shape:", images.shape)
    print("Kernel shape:", kernel.shape)
    print("Convolved output shape:", output.shape)
    print("Output:\n", output)
