#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network"""

    def __init__(self, nx, layers):
        """Initialize deep neural network"""

        # ---------------- VALIDATION ORDER (IMPORTANT) ----------------

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        # ---------------- ATTRIBUTES ----------------
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # ---------------- INITIALIZATION ----------------
        for l in range(1, self.L + 1):

            if l == 1:
                self.weights["W1"] = (
                    np.random.randn(layers[0], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                self.weights["W{}".format(l)] = (
                    np.random.randn(
                        layers[l - 1],
                        layers[l - 2]
                    ) * np.sqrt(2 / layers[l - 2])
                )

            self.weights["b{}".format(l)] = np.zeros((layers[l - 1], 1))
