#!/usr/bin/env python3
"""L2 Regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
    cost -- cost without regularization
    lambtha -- regularization parameter
    weights -- dictionary containing weights and biases
    L -- number of layers
    m -- number of data points

    Returns:
    Cost with L2 regularization
    """
    l2_sum = 0

    for i in range(1, L + 1):
        l2_sum += np.sum(np.square(weights["W{}".format(i)]))

    return cost + (lambtha / (2 * m)) * l2_sum
