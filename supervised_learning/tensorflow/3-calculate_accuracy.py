#!/usr/bin/env python3
"""
Calculates the accuracy of a prediction
"""

import tensorflow as tf  # type: ignore


def calculate_accuracy(y, y_pred):
    """
    y: placeholder for true labels
    y_pred: predicted tensor

    Returns:
        tensor representing accuracy
    """

    correct_predictions = tf.equal(
        tf.argmax(y, axis=1),
        tf.argmax(y_pred, axis=1)
    )

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy