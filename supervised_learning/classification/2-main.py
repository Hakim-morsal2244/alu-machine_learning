#!/usr/bin/env python3

import numpy as np

Neuron = __import__('2-neuron').Neuron

np.random.seed(0)

X = np.random.randn(5, 3)

neuron = Neuron(5)

A = neuron.forward_prop(X)

print(A)