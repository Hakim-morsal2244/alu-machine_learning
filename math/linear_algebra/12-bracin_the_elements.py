#!/usr/bin/env python3
"""
Module that contains a function to perform element-wise arithmetic operations.
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication,
    and division of two numpy arrays.

    Returns:
        tuple: (add, sub, mul, div)
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
