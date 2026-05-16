#!/usr/bin/env python3
"""Module defines a Neuron for binary classification"""

import numpy as np


class Neuron:
    """Neuron performing binary classification"""

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
        """Weights getter"""
        return self.__W

    @property
    def b(self):
        """Bias getter"""
        return self.__b

    @property
    def A(self):
        """Activation getter"""
        return self.__A