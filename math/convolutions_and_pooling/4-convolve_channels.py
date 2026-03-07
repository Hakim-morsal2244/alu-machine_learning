#!/usr/bin/env python3
"""Performs a convolution on multi-channel images."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Parameters
    ----------
    images : numpy.ndarray
        (m, h, w, c) array of m images with c channels
    kernel : numpy.ndarray
        (kh, kw, c) convolution kernel
    padding : str or tuple
        'same', 'valid', or (ph, pw)
    stride : tuple
        (sh, sw) stride for height and width

    Returns
    -------
    numpy.ndarray
        Convolved images of shape (m, out_h, out_w)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channel dimension must match image channels")

    # Determine padding
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images (only height and width, not channels)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    out_h = (padded.shape[1] - kh) // sh + 1
    out_w = (padded.shape[2] - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w))

    # Convolve using two loops (i over height, j over width)
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            region = padded[:, h_start:h_end, w_start:w_end, :]
            output[:, i, j] = np.sum(
                region * kernel[np.newaxis, :, :, :],
                axis=(1, 2, 3)
            )

    return output
