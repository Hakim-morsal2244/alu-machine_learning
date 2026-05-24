#!/usr/bin/env python3
"""
Creates a neural network layer using TensorFlow
"""

import tensorflow as tf  # type: ignore


def create_layer(prev, n, activation):
    """
    prev: tensor output of previous layer
    n: number of nodes in the layer
    activation: activation function

    Returns:
        tensor output of the layer
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )

    return layer(prev)
