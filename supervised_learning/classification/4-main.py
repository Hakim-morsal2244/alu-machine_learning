#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

lib = np.load('../data/Binary_Train.npz')
X = lib['X'].reshape((lib['X'].shape[0], -1)).T
Y = lib['Y']

np.random.seed(0)
neuron = Neuron(X.shape[0])

A, cost = neuron.evaluate(X, Y)

print(A)
print(cost)