#!/usr/bin/env python3
"""Performs convolution on images using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters
    ----------
    images : numpy.ndarray
        (m, h, w, c) array of images
    kernels : numpy.ndarray
        (kh, kw, c, nc) array of kernels
    padding : str or tuple
        'same', 'valid', or (ph, pw)
    stride : tuple
        (sh, sw) stride for height and width

    Returns
    -------
    numpy.ndarray
        Convolved images of shape (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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

    # Pad images (only height and width)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = (padded.shape[1] - kh) // sh + 1
    out_w = (padded.shape[2] - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w, nc))

    # Three loops: over i, j, and kernel index k
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            region = padded[:, h_start:h_end, w_start:w_end, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    region * kernels[np.newaxis, :, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
