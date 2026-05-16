#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network"""

    def __init__(self, nx, layers):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (not isinstance(layers, list) or len(layers) == 0):
            raise TypeError(
                "layers must be a list of positive integers"
            )

        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError(
                "layers must be a list of positive integers"
            )

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            nodes = layers[i]

            if i == 0:
                self.weights["W1"] = (
                    np.random.randn(nodes, nx)
                    * np.sqrt(2 / nx)
                )
            else:
                self.weights["W{}".format(i + 1)] = (
                    np.random.randn(nodes, layers[i - 1])
                    * np.sqrt(2 / layers[i - 1])
                )

            self.weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):

        self.cache["A0"] = X

        for i in range(1, self.L + 1):

            W = self.weights["W{}".format(i)]
            b = self.weights["b{}".format(i)]

            Z = np.matmul(W, self.cache["A{}".format(i - 1)]) + b
            A = self.sigmoid(Z)

            self.cache["A{}".format(i)] = A

        return A, self.cache

    def gradient_descent(self, Y, cache, alpha=0.05):

        m = Y.shape[1]
        L = self.L

        A_L = cache["A{}".format(L)]
        dZ = A_L - Y

        for i in range(L, 0, -1):

            A_prev = cache["A{}".format(i - 1)]
            W = self.weights["W{}".format(i)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.weights["W{}".format(i)] -= alpha * dW
            self.weights["b{}".format(i)] -= alpha * db

            if i > 1:
                A_prev_prev = cache["A{}".format(i - 1)]
                dZ = np.matmul(W.T, dZ) * (A_prev_prev * (1 - A_prev_prev))