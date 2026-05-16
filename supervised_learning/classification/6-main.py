#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('6-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train = lib_train['X'].reshape((lib_train['X'].shape[0], -1)).T
Y_train = lib_train['Y']

lib_dev = np.load('../data/Binary_Dev.npz')
X_dev = lib_dev['X'].reshape((lib_dev['X'].shape[0], -1)).T
Y_dev = lib_dev['Y']

np.random.seed(0)

neuron = Neuron(X_train.shape[0])

A, cost = neuron.train(X_train, Y_train, iterations=10)

print("Train cost:", cost)
print("Train data:", A)
print("Train Neuron A:", neuron.A)

A_dev, cost_dev = neuron.evaluate(X_dev, Y_dev)

print("Dev cost:", cost_dev)
print("Dev data:", A_dev)
print("Dev Neuron A:", neuron.A)

# -------------------------------
# SAFE VISUALIZATION (NO reshape errors)
# -------------------------------

fig = plt.figure(figsize=(10, 10))

for i in range(100):
    fig.add_subplot(10, 10, i + 1)

    # safe reshape (works for ANY square-like feature set)
    size = int(np.sqrt(X_dev.shape[0]))

    if size * size == X_dev.shape[0]:
        img = X_dev[:, i].reshape(size, size)
    else:
        img = X_dev[:, i].reshape(1, -1)

    plt.imshow(img)
    plt.title(str(A_dev[0, i]))
    plt.axis('off')

plt.tight_layout()
plt.show()