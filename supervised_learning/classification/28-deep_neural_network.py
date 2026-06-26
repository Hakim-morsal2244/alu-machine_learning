#!/usr/bin/env python3
"""Deep Neural Network"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network"""

    def __init__(self, nx, layers, activation='sig'):
        """class constructor"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (not isinstance(layers, list) or
                len(layers) == 0):
            raise TypeError(
                "layers must be a list of positive integers"
            )

        if activation not in ['sig', 'tanh']:
            raise ValueError(
                "activation must be 'sig' or 'tanh'"
            )

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):

            if (not isinstance(layers[i], int)
                    or layers[i] <= 0):
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            if i == 0:
                prev = nx
            else:
                prev = layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(layers[i], prev)
                * np.sqrt(2 / prev)
            )

            self.__weights["b{}".format(i + 1)] = np.zeros(
                (layers[i], 1)
            )

    @property
    def L(self):
        """getter"""
        return self.__L

    @property
    def cache(self):
        """getter"""
        return self.__cache

    @property
    def weights(self):
        """getter"""
        return self.__weights

    @property
    def activation(self):
        """getter"""
        return self.__activation

    def forward_prop(self, X):
        """forward propagation"""

        self.__cache["A0"] = X

        for i in range(self.__L):

            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]

            Z = np.matmul(
                W,
                self.__cache["A{}".format(i)]
            ) + b

            if i != self.__L - 1:

                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)

            else:
                A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """cost function"""

        m = Y.shape[1]

        cost = -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m

        return cost

    def evaluate(self, X, Y):
        """evaluate"""

        A, _ = self.forward_prop(X)

        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient descent"""

        m = Y.shape[1]

        weights_copy = self.__weights.copy()

        dZ = cache["A{}".format(self.__L)] - Y

        for i in reversed(range(self.__L)):

            A_prev = cache["A{}".format(i)]

            W = weights_copy["W{}".format(i + 1)]

            dW = np.matmul(dZ, A_prev.T) / m

            db = np.sum(
                dZ,
                axis=1,
                keepdims=True
            ) / m

            self.__weights["W{}".format(i + 1)] = (
                self.__weights["W{}".format(i + 1)]
                - alpha * dW
            )

            self.__weights["b{}".format(i + 1)] = (
                self.__weights["b{}".format(i + 1)]
                - alpha * db
            )

            if i != 0:

                if self.__activation == 'sig':
                    dZ = np.matmul(
                        W.T,
                        dZ
                    ) * (
                        A_prev * (1 - A_prev)
                    )

                else:
                    dZ = np.matmul(
                        W.T,
                        dZ
                    ) * (
                        1 - (A_prev ** 2)
                    )

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True,
              graph=True, step=100):
        """train network"""

        if not isinstance(iterations, int):
            raise TypeError(
                "iterations must be an integer"
            )

        if iterations <= 0:
            raise ValueError(
                "iterations must be a positive integer"
            )

        if not isinstance(alpha, float):
            raise TypeError(
                "alpha must be a float"
            )

        if alpha <= 0:
            raise ValueError(
                "alpha must be positive"
            )

        for i in range(iterations):

            A, cache = self.forward_prop(X)

            self.gradient_descent(
                Y,
                cache,
                alpha
            )

        return self.evaluate(X, Y)

    def save(self, filename):
        """save object"""

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """load object"""

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)

        except FileNotFoundError:
            return None
