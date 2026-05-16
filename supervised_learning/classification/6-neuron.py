#!/usr/bin/env python3
"""Neuron class with training"""

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
        Forward propagation

        Args:
            X (numpy.ndarray): input data (nx, m)

        Returns:
            numpy.ndarray: activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Computes cost

        Args:
            Y (numpy.ndarray): correct labels
            A (numpy.ndarray): activated output

        Returns:
            float: cost
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """
        Evaluates neuron

        Returns:
            tuple: (predictions, cost)
        """
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Gradient descent step
        """
        m = Y.shape[1]

        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains neuron

        Args:
            X (numpy.ndarray): input data
            Y (numpy.ndarray): labels
            iterations (int): training steps
            alpha (float): learning rate

        Returns:
            tuple: (predictions, cost)
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
