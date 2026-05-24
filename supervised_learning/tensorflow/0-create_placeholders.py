#!/usr/bin/env python3
"""Creates TensorFlow placeholders"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """Creates placeholders for input data and labels"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
