#!/usr/bin/env python3
"""Creates TensorFlow placeholders"""

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def create_placeholders(nx, classes):
    """Creates placeholders for input data and labels"""
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
