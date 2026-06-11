#!/usr/bin/env python3
"""Calculates specificity for each class in a confusion matrix."""

import numpy as np


def specificity(confusion):
    """
    Calculates specificity for each class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    classes = confusion.shape[0]
    total = np.sum(confusion, axis=1).sum()
    tn_fp_fn_tp = total

    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = tn_fp_fn_tp - (tp + fp + fn)

    return tn / (tn + fp)
