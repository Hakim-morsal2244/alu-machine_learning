#!/usr/bin/env python3
"""Adam upgraded"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network
    using Adam optimization

    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: weight for first moment
        beta2: weight for second moment
        epsilon: small number to avoid division by zero

    Returns:
        Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss)

    return train_op
