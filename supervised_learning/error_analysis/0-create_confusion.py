#!/usr/bin/env python3
"""Creates a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
        logits: one-hot numpy.ndarray of shape (m, classes)

    Returns:
        numpy.ndarray of shape (classes, classes)
    """
    classes = labels.shape[1]

    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    confusion = np.zeros((classes, classes))

    for i in range(labels.shape[0]):
        confusion[true[i], pred[i]] += 1

    return confusion
