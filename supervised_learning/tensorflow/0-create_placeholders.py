#!/usr/bin/env python3
"""
Creates TensorFlow placeholders
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    nx: number of input features
    classes: number of classes

    Returns:
        x, y placeholders
    """

    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
