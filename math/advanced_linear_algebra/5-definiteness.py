#!/usr/bin/env python3

"""
5-definiteness.py
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    Returns one of:
    'Positive definite', 'Positive semi-definite',
    'Negative semi-definite', 'Negative definite', 'Indefinite',
    or None if not applicable.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # check if matrix is square and non-empty
    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        # compute eigenvalues
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0) and not pos
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0) and not neg
    indefinite = not (pos or pos_semi or neg or neg_semi)

    if pos:
        return "Positive definite"
    elif pos_semi:
        return "Positive semi-definite"
    elif neg:
        return "Negative definite"
    elif neg_semi:
        return "Negative semi-definite"
    elif indefinite:
        return "Indefinite"

    return None
