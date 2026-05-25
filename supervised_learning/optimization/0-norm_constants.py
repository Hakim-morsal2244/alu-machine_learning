#!/usr/bin/env python3
"""Normalization constants module"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization constants of a matrix

    Args:
        X: numpy.ndarray of shape (m, nx)

    Returns:
        mean, standard deviation
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
