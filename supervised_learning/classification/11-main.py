#!/usr/bin/env python3

import numpy as np

NN = __import__('11-neural_network').NeuralNetwork

lib = np.load('../data/Binary_Train.npz')
X = lib['X'].reshape((lib['X'].shape[0], -1)).T
Y = lib['Y']

np.random.seed(0)

nn = NN(X.shape[0], 3)

_, A = nn.forward_prop(X)
cost = nn.cost(Y, A)

print(cost)