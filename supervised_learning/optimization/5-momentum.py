#!/usr/bin/env python3
"""Momentum optimization update"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum

    alpha: learning rate
    beta1: momentum weight
    var: numpy array (parameter)
    grad: gradient of var
    v: previous first moment

    Returns: updated var, new v
    """

    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new

    return var_new, v_new
