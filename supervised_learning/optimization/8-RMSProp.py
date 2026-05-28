#!/usr/bin/env python3
"""RMSProp upgraded"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    using RMSProp optimization

    Args:
        loss: loss of the network
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero

    Returns:
        RMSProp optimization operation
    """

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss)

    return train_op
