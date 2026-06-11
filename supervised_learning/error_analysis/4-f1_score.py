#!/usr/bin/env python3
"""Calculates F1 score for each class in a confusion matrix."""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates F1 score per class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    rec = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * rec) / (prec + rec)
