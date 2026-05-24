#!/usr/bin/env python3
"""
Creates training operation using gradient descent
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss: loss tensor
    alpha: learning rate

    Returns:
        training operation
    """

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)

    return train_op
