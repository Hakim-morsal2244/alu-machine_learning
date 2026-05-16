#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (not isinstance(layers, list) or len(layers) == 0):
            raise TypeError(
                "layers must be a list of positive integers"
            )

        if not all(
            isinstance(x, int) and x > 0 for x in layers
        ):
            raise TypeError(
                "layers must be a list of positive integers"
            )

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer_idx in range(self.L):
            layer_size = layers[layer_idx]

            if layer_idx == 0:
                self.weights["W1"] = (
                    np.random.randn(layer_size, nx)
                    * np.sqrt(2 / nx)
                )
            else:
                self.weights["W{}".format(layer_idx + 1)] = (
                    np.random.randn(
                        layer_size,
                        layers[layer_idx - 1]
                    ) * np.sqrt(2 / layers[layer_idx - 1])
                )

            self.weights["b{}".format(layer_idx + 1)] = np.zeros(
                (layer_size, 1)
            )

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Forward propagation"""

        self.cache["A0"] = X

        for layer in range(1, self.L + 1):

            W = self.weights["W{}".format(layer)]
            b = self.weights["b{}".format(layer)]

            Z = np.matmul(W, self.cache["A{}".format(layer - 1)]) + b
            A = self.sigmoid(Z)

            self.cache["A{}".format(layer)] = A

        return A, self.cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One step of gradient descent"""

        m = Y.shape[1]
        L = self.L

        A_L = cache["A{}".format(L)]
        dZ = A_L - Y

        for layer in reversed(range(1, L + 1)):

            A_prev = cache["A{}".format(layer - 1)]

            W = self.weights["W{}".format(layer)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer > 1:
                dA_prev = np.matmul(W.T, dZ)
                dZ = dA_prev * (cache["A{}".format(layer - 1)] *
                                (1 - cache["A{}".format(layer - 1)]))

            self.weights["W{}".format(layer)] -= alpha * dW
            self.weights["b{}".format(layer)] -= alpha * db
