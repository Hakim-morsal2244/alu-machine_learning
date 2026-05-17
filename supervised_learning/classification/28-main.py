#!/usr/bin/env python3
"""Main file for testing Task 28"""

import numpy as np

Deep = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

# Load MNIST dataset
lib = np.load('../data/MNIST.npz')

X_train_3D = lib['X_train']
Y_train = lib['Y_train']

X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']

# Reshape data
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T

# One-hot encode labels
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)

print("Sigmoid activation:")

# Create model directly instead of loading missing pkl
deep27 = Deep(X_train.shape[0], [128, 64, 10])

# Train briefly for testing
A_one_hot27, cost27 = deep27.train(
    X_train,
    Y_train_one_hot,
    iterations=10,
    alpha=0.05,
    verbose=False,
    graph=False
)

# Evaluate
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_train == A27) / Y_train.shape[0] * 100

print("Train cost:", cost27)
print("Train accuracy: {}%".format(accuracy27))

A_valid, cost_valid = deep27.evaluate(X_valid, Y_valid_one_hot)
A_valid = one_hot_decode(A_valid)

accuracy_valid = np.sum(Y_valid == A_valid) / Y_valid.shape[0] * 100

print("Validation cost:", cost_valid)
print("Validation accuracy: {}%".format(accuracy_valid))