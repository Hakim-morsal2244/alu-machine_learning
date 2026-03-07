#!/usr/bin/env python3
"""Performs strided convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with stride.

    Parameters
    ----------
    images : numpy.ndarray
        (m, h, w) array of m grayscale images
    kernel : numpy.ndarray
        (kh, kw) convolution kernel
    padding : str or tuple
        'same', 'valid', or (ph, pw)
    stride : tuple
        (sh, sw) stride for height and width

    Returns
    -------
    numpy.ndarray
        Convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    out_h = (padded.shape[1] - kh) // sh + 1
    out_w = (padded.shape[2] - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w))

    # Convolve using two loops
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            region = padded[:, h_start:h_end, w_start:w_end]
            output[:, i, j] = np.sum(
                region * kernel[np.newaxis, :, :],
                axis=(1, 2)
            )

    return output
