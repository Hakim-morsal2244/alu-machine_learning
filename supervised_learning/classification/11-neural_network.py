#!/usr/bin/env python3
"""Neural Network with one hidden layer (cost function added)"""

import numpy as np


class NeuralNetwork:
    """Neural network for binary classification"""

    def __init__(self, nx, nodes):
        """Initialize neural network"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # ---------------- GETTERS ----------------

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    # ---------------- FORWARD PROP ----------------

    def forward_prop(self, X):
        """Forward propagation"""

        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    # ---------------- COST FUNCTION ----------------

    def cost(self, Y, A):
        """
        Logistic regression cost function

        Args:
            Y (numpy.ndarray): correct labels (1, m)
            A (numpy.ndarray): predictions (1, m)

        Returns:
            float: cost
        """

        m = Y.shape[1]

        cost = -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m

        return cost
