#!/usr/bin/env python3

import numpy as np

NN = __import__('12-neural_network').NeuralNetwork

lib = np.load('../data/Binary_Train.npz')
X = lib['X'].reshape((lib['X'].shape[0], -1)).T
Y = lib['Y']

np.random.seed(0)

nn = NN(X.shape[0], 3)

A, cost = nn.evaluate(X, Y)

print(A)
print(cost)