#!/usr/bin/env python3
"""
Module that contains a function to concatenate two numpy arrays.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays along the specified axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
