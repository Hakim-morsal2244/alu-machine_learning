#!/usr/bin/env python3
"""Gradient descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Performs gradient descent with dropout

    Y: one-hot labels (classes, m)
    weights: dictionary of W and b (updated in place)
    cache: forward propagation cache including dropout masks
    alpha: learning rate
    keep_prob: dropout probability
    L: number of layers
    """

    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(layer - 1)]

        W = weights["W{}".format(layer)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # L2 not used here (dropout only task)

        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)

            D = cache["D{}".format(layer - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob

            dZ = dA_prev * (1 - np.power(cache["A{}".format(layer - 1)], 2))
