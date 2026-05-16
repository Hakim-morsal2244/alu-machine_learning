#!/usr/bin/env python3

import numpy as np

Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

lib = np.load('../data/Binary_Train.npz')
X = lib['X'].reshape((lib['X'].shape[0], -1)).T
Y = lib['Y']

np.random.seed(0)

deep = Deep(X.shape[0], [5, 3, 1])

print(deep.cache)
print(deep.weights)
print(deep.L)

deep.L = 10
print(deep.L)