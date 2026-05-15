#!/usr/bin/env python3

import numpy as np

Neuron = __import__('0-neuron').Neuron

np.random.seed(0)
neuron = Neuron(5)

print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)