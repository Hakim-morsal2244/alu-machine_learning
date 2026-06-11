#!/usr/bin/env python3
"""Calculates sensitivity for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates sensitivity (recall) for each class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    tp = np.diag(confusion)
    fn_plus_tp = np.sum(confusion, axis=1)

    # avoid division by zero
    return tp / fn_plus_tp
