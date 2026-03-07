#!/usr/bin/env python3
"""Performs a convolution on grayscale images with stride."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            region = padded[
                :, h_start:h_end, w_start:w_end
            ]

            output[:, i, j] = np.sum(
                region * kernel[np.newaxis, :, :],
                axis=(1, 2)
            )

    return output
