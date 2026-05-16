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

        for x in layers:
            if not isinstance(x, int) or x <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        layer = 1

        self.weights["W1"] = (
            np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
        )
        self.weights["b1"] = np.zeros((layers[0], 1))

        while layer < self.L:
            self.weights["W{}".format(layer + 1)] = (
                np.random.randn(
                    layers[layer],
                    layers[layer - 1]
                ) * np.sqrt(2 / layers[layer - 1])
            )

            self.weights["b{}".format(layer + 1)] = np.zeros(
                (layers[layer], 1)
            )

            layer += 1
