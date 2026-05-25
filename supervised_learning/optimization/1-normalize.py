#!/usr/bin/env python3
"""Normalize module"""

import numpy as np


def normalize(X, m, s):
    """
    normalizes a matrix

    Args:
        X: numpy.ndarray of shape (d, nx)
        m: mean of each feature
        s: standard deviation of each feature

    Returns:
        normalized matrix
    """
    return (X - m) / s
