#!/usr/bin/env python3
"""
Forward propagation graph for a neural network
"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x: input placeholder
    layer_sizes: list of number of nodes per layer
    activations: list of activation functions

    Returns:
        prediction tensor of the network
    """

    A = x

    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])

    return A
