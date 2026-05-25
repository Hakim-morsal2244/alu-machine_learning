#!/usr/bin/env python3
"""Shuffle data module"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffles two matrices the same way

    Args:
        X: numpy.ndarray of shape (m, nx)
        Y: numpy.ndarray of shape (m, ny)

    Returns:
        shuffled X and Y
    """
    permutation = np.random.permutation(X.shape[0])

    return X[permutation], Y[permutation]
