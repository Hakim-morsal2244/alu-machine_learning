#!/usr/bin/env python3

import numpy as np

NN = __import__('9-neural_network').NeuralNetwork

lib = np.load('../data/Binary_Train.npz')
X = lib['X'].reshape((lib['X'].shape[0], -1)).T

np.random.seed(0)

nn = NN(X.shape[0], 3)

print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
print(nn.A1)
print(nn.A2)

nn.A1 = 10
print(nn.A1)