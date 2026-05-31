#!/usr/bin/env python3
"""
Module that creates a neural network layer with Dropout regularization.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: tensor of shape (m, nx) - output of previous layer
        n: number of nodes in the new layer
        activation: activation function for the layer
        keep_prob: probability that a node will be kept

    Returns:
        Output tensor of the new layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )(prev)

    # Apply dropout AFTER activation
    dropout = tf.layers.Dropout(rate=1 - keep_prob)(layer)

    return dropout
