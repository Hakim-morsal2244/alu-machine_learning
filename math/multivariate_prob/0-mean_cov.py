#!/usr/bin/env python3
"""
Module that calculates the mean and covariance of a dataset
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a dataset

    Parameters:
    X (numpy.ndarray): shape (n, d)

    Returns:
    mean (numpy.ndarray): shape (1, d)
    cov (numpy.ndarray): shape (d, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n = X.shape[0]

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    X_centered = X - mean

    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
