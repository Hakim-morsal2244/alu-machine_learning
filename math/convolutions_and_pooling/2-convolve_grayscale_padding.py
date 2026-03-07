#!/usr/bin/env python3
"""Perform convolution on grayscale images with custom padding."""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform convolution on grayscale images with custom padding.

    Parameters
    ----------
    images : numpy.ndarray
        Numpy array of shape (m, h, w) containing m grayscale images
    kernel : numpy.ndarray
        Numpy array of shape (kh, kw) containing the kernel
    padding : tuple
        Tuple of (ph, pw) specifying the padding for height and width

    Returns
    -------
    numpy.ndarray
        Numpy array containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Output dimensions after padding
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    # Perform convolution using only two loops
    for i in range(output_h):
        for j in range(output_w):
            # Slice the padded images
            image_region = padded_images[:, i:i + kh, j:j + kw]
            # Element-wise multiply and sum
            output[:, i, j] = np.sum(image_region * kernel, axis=(1, 2))

    return output
