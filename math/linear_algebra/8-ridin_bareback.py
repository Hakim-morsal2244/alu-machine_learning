#!/usr/bin/env python3
"""
Module that contains a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiply two 2D matrices (mat1 * mat2).

    Args:
        mat1 (list of list of int/float): First matrix.
        mat2 (list of list of int/float): Second matrix.

    Returns:
        list: New matrix with multiplication result.
        None: If matrices cannot be multiplied.
    """
    # Check if multiplication is possible
    if len(mat1[0]) != len(mat2):
        return None

    # Create result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
