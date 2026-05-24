#!/usr/bin/env python3
"""
Module that contains a function that returns two placeholders, x and y
"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates placeholders for input data and labels

    nx: number of input features
    classes: number of classes

    Returns:
        x: input placeholder
        y: label placeholder
    """

    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
