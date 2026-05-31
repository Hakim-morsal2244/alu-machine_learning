#!/usr/bin/env python3
"""Learning rate decay operation in TensorFlow"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation using inverse time decay

    Args:
        alpha: original learning rate
        decay_rate: decay rate
        global_step: number of gradient descent passes elapsed
        decay_step: number of passes before decay

    Returns:
        learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )
