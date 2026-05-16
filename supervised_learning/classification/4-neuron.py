#!/usr/bin/env python3
"""Neuron class with evaluation"""

import numpy as np


class Neuron:
    """Neuron for binary classification"""

    def __init__(self, nx):
        """Initialize neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W"""
        return self.__W

    @property
    def b(self):
        """Getter for b"""
        return self.__b

    @property
    def A(self):
        """Getter for A"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron

        Args:
            X (numpy.ndarray): input data (nx, m)

        Returns:
            numpy.ndarray: activated output (A)
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression

        Args:
            Y (numpy.ndarray): correct labels (1, m)
            A (numpy.ndarray): activated output (1, m)

        Returns:
            float: cost value
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """
        Evaluates the neuron

        Args:
            X (numpy.ndarray): input data (nx, m)
            Y (numpy.ndarray): correct labels (1, m)

        Returns:
            tuple: (predictions, cost)
        """
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost
