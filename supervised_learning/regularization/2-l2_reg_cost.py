#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization

    cost: tensor containing the base cost (without regularization)

    Returns:
        tensor containing cost including L2 regularization
    """
    l2_cost = tf.losses.get_regularization_loss()
    return cost + l2_cost
