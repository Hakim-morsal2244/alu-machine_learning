#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network"""

    def __init__(self, nx, layers):
        """Initialize network"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
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

        for layer in range(self.L):

            if layer == 0:
                self.weights["W1"] = (
                    np.random.randn(layers[0], nx)
                    * np.sqrt(2 / nx)
                )
            else:
                self.weights["W{}".format(layer + 1)] = (
                    np.random.randn(
                        layers[layer],
                        layers[layer - 1]
                    ) * np.sqrt(2 / layers[layer - 1])
                )

            self.weights["b{}".format(layer + 1)] = np.zeros(
                (layers[layer], 1)
            )
