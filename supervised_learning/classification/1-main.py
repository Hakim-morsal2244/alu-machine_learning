#!/usr/bin/env python3

import numpy as np

Neuron = __import__('1-neuron').Neuron

np.random.seed(0)

neuron = Neuron(5)

print(neuron.W)
print(neuron.b)
print(neuron.A)

try:
    neuron.A = 10
except Exception as e:
    print(e)