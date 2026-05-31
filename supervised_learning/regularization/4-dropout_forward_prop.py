#!/usr/bin/env python3
"""Forward propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with dropout

    X: input data (nx, m)
    weights: dictionary of W and b
    L: number of layers
    keep_prob: probability a node is kept

    Returns:
        cache: dictionary of activations and dropout masks
    """

    cache = {}
    cache["A0"] = X

    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]

        Z = np.matmul(W, cache["A{}".format(layer - 1)]) + b

        # Last layer -> softmax
        if layer == L:
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = A * D
            A = A / keep_prob
            cache["D{}".format(layer)] = D

        cache["A{}".format(layer)] = A

    return cache
