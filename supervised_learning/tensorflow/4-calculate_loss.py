#!/usr/bin/env python3
"""
Calculates softmax cross-entropy loss
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    y: placeholder for true labels
    y_pred: predicted tensor (logits)

    Returns:
        tensor representing softmax cross-entropy loss
    """

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )

    return loss
