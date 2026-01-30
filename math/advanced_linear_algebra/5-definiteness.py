#!/usr/bin/env python3

"""
5-definiteness.py
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a symmetric matrix.
    Returns one of:
    'Positive definite', 'Positive semi-definite',
    'Negative semi-definite', 'Negative definite', 'Indefinite',
    or None if the matrix is not valid or not symmetric.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # must be square and non-empty
    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        return None

    # must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    if np.all(eigvals > 0):
        return "Positive definite"
    elif np.all(eigvals >= 0):
        return "Positive semi-definite"
    elif np.all(eigvals < 0):
        return "Negative definite"
    elif np.all(eigvals <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
