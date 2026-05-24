#!/usr/bin/env python3
"""Creates TensorFlow placeholders"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """
    Returns two placeholders for the neural network

    nx: number of feature columns
    classes: number of classes

    Returns: x, y
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y