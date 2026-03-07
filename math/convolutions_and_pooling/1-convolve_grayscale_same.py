#!/usr/bin/env python3
"""Perform same convolution on grayscale images using a kernel."""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images.

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

    # Calculate padding for height and width
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    # Initialize output array (same shape as input)
    output = np.zeros((m, h, w))

    # Perform convolution using only two loops
    for i in range(h):
        for j in range(w):
            # Slice the padded images
            image_region = padded_images[:, i:i + kh, j:j + kw]
            # Element-wise multiply and sum
            output[:, i, j] = np.sum(image_region * kernel, axis=(1, 2))

    return output
