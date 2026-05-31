#!/usr/bin/env python3
"""Gradient descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights using gradient descent with dropout
    """

    m = Y.shape[1]

    # ----- Output layer gradient (softmax) -----
    A_L = cache["A{}".format(L)]
    dZ = A_L - Y

    for layer in reversed(range(1, L + 1)):

        A_prev = cache["A{}".format(layer - 1)]

        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]

        # gradients
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # update weights
        weights["W{}".format(layer)] = W - alpha * dW
        weights["b{}".format(layer)] = b - alpha * db

        if layer > 1:
            W_curr = W

            dA_prev = np.matmul(W_curr.T, dZ)

            # Apply dropout mask (IMPORTANT: from SAME layer)
            D = cache["D{}".format(layer - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob

            # tanh derivative
            A_prev_act = cache["A{}".format(layer - 1)]
            dZ = dA_prev * (1 - np.square(A_prev_act))
