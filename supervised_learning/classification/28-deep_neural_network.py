#!/usr/bin/env python3
"""Deep Neural Network with sigmoid/tanh activation"""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network for binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            prev = nx if i == 0 else layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    # ---------------- Properties ----------------
    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    # ---------------- Forward Prop ----------------
    def forward_prop(self, X):
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]

            Z = W @ self.__cache["A{}".format(i - 1)] + b

            if self.__activation == "sig":
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    # ---------------- Cost ----------------
    def cost(self, Y, A):
        m = Y.shape[1]
        A = np.clip(A, 1e-8, 1 - 1e-8)
        return -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1 - A)
        ) / m

    # ---------------- Evaluate ----------------
    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, self.cost(Y, A)

    # ---------------- Gradient Descent ----------------
    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L

        dZ = cache["A{}".format(L)] - Y

        for i in reversed(range(1, L + 1)):
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]

            dW = (1 / m) * (dZ @ A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 1:
                if self.__activation == "sig":
                    dZ = (W.T @ dZ) * (A_prev * (1 - A_prev))
                else:
                    dZ = (W.T @ dZ) * (1 - A_prev ** 2)

            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db

        return self.__weights

    # ---------------- Train ----------------
    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=False, step=100):

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)

            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            if verbose and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(
                    i, self.cost(Y, A)
                ))

        return self.evaluate(X, Y)

    # ---------------- Save / Load ----------------
    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
