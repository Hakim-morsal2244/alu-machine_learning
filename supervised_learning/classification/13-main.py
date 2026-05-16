#!/usr/bin/env python3

import numpy as np

NN = __import__('13-neural_network').NeuralNetwork

lib = np.load('../data/Binary_Train.npz')
X = lib['X'].reshape((lib['X'].shape[0], -1)).T
Y = lib['Y']

np.random.seed(0)

nn = NN(X.shape[0], 3)

A1, A2 = nn.forward_prop(X)

nn.gradient_descent(X, Y, A1, A2, 0.5)

print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)