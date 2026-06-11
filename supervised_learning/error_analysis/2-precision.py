#!/usr/bin/env python3
"""Calculates precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """
    Calculates precision for each class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    tp = np.diag(confusion)
    fp_plus_tp = np.sum(confusion, axis=0)

    return tp / fp_plus_tp
