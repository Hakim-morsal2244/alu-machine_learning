#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network"""

    def __init__(self, nx, layers):
        """Initialize the network"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (not isinstance(layers, list) or len(layers) == 0):
            raise TypeError(
                "layers must be a list of positive integers"
            )

        for x in layers:
            if not isinstance(x, int) or x <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer in range(1, self.L + 1):

            if layer == 1:
                rand = np.random.randn(layers[0], nx)
                self.weights["W1"] = rand * np.sqrt(2 / nx)
            else:
                rand = np.random.randn(
                    layers[layer - 1],
                    layers[layer - 2]
                )
                self.weights["W{}".format(layer)] = (
                    rand * np.sqrt(2 / layers[layer - 2])
                )

            self.weights["b{}".format(layer)] = np.zeros(
                (layers[layer - 1], 1)
            )
