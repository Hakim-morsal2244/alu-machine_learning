#!/usr/bin/env python3
"""Batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n)
        gamma: numpy.ndarray of shape (1, n)
        beta: numpy.ndarray of shape (1, n)
        epsilon: small number to avoid division by zero

    Returns:
        normalized Z matrix
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
