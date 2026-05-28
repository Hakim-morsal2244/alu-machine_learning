#!/usr/bin/env python3
"""Adam optimization"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    updates a variable using Adam optimization algorithm

    Args:
        alpha: learning rate
        beta1: weight for first moment
        beta2: weight for second moment
        epsilon: small number to avoid division by zero
        var: variable to update
        grad: gradient of var
        v: previous first moment
        s: previous second moment
        t: time step for bias correction

    Returns:
        updated variable, new first moment, new second moment
    """

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - (beta1 ** t))
    s_corrected = s / (1 - (beta2 ** t))

    var = var - alpha * (
        v_corrected / (np.sqrt(s_corrected) + epsilon)
    )

    return var, v, s
