#!/usr/bin/env python3
"""Perform strided convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform a strided convolution on grayscale images.

    Parameters
    ----------
    images : numpy.ndarray
        Numpy array of shape (m, h, w) with m grayscale images
    kernel : numpy.ndarray
        Numpy array of shape (kh, kw) with the convolution kernel
    padding : str or tuple
        'same', 'valid', or tuple (ph, pw) for custom padding
    stride : tuple
        Tuple (sh, sw) specifying the stride for height and width

    Returns
    -------
    numpy.ndarray
        Numpy array containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Only pad if needed
    if ph > 0 or pw > 0:
        padded_images = np.pad(
            images,
            pad_width=((0, 0), (ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_images = images

    # Compute output dimensions
    out_h = (padded_images.shape[1] - kh) // sh + 1
    out_w = (padded_images.shape[2] - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Convolution using two loops
    for i in range(out_h):
        for j in range(out_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            region = padded_images[
                :, vert_start:vert_end, horiz_start:horiz_end
            ]
            output[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
            )

    return output
