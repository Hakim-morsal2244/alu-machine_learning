#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates cost including L2 regularization term per layer

    cost: tensor (base cost)

    Returns:
        tensor: cost + L2 regularization term
    """
    l2_costs = tf.losses.get_regularization_losses()
    return cost + l2_costs
