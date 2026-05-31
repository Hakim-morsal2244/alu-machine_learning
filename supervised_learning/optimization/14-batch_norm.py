#!/usr/bin/env python3
"""Batch normalization layer"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer

    Args:
        prev: activated output of previous layer
        n: number of nodes in layer
        activation: activation function

    Returns:
        activated output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=init
    )

    Z = dense(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])

    gamma = tf.Variable(
        tf.ones([n]),
        trainable=True
    )

    beta = tf.Variable(
        tf.zeros([n]),
        trainable=True
    )

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-8
    )

    return activation(Z_norm)