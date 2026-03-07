#!/usr/bin/env python3
"""Performs pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters
    ----------
    images : numpy.ndarray
        (m, h, w, c) array of images
    kernel_shape : tuple
        (kh, kw) pooling window
    stride : tuple
        (sh, sw) stride for pooling
    mode : str
        'max' or 'avg'

    Returns
    -------
    numpy.ndarray
        Pooled images of shape (m, out_h, out_w, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            region = images[:, h_start:h_end, w_start:w_end, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("Mode must be 'max' or 'avg'")

    return output
