#!/usr/bin/env python3
"""L2 Regularization Gradient Descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization

    Y: one-hot labels
    weights: dictionary containing weights and biases
    cache: dictionary containing activations
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers
    """
    m = Y.shape[1]

    dZ = cache["A{}".format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]
        W = weights["W{}".format(layer)]

        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            A_prev_layer = cache["A{}".format(layer - 1)]
            dZ_prev = np.matmul(W.T, dZ)
            dZ_prev *= (1 - A_prev_layer ** 2)

        weights["W{}".format(layer)] = W - alpha * dW
        weights["b{}".format(layer)] = (
            weights["b{}".format(layer)] - alpha * db
        )

        if layer > 1:
            dZ = dZ_prev
