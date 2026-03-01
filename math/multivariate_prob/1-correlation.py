#!/usr/bin/env python3
"""
Module that calculates a correlation matrix from a covariance matrix
"""
import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix

    Parameters:
    C (numpy.ndarray): shape (d, d) covariance matrix

    Returns:
    numpy.ndarray: shape (d, d) correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    stddev = np.sqrt(np.diag(C))  # Standard deviations

    # Avoid division by zero
    if np.any(stddev == 0):
        raise ValueError("Covariance matrix has zero variance on diagonal")

    # Outer product of stddev
    denom = np.outer(stddev, stddev)
    corr = C / denom

    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr, 1.0)

    return corr
