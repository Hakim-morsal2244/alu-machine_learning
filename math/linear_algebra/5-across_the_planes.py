#!/usr/bin/env python3
"""
Module that contains function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): First matrix.
        mat2 (list of list of int/float): Second matrix.

    Returns:
        list of list of int/float: New matrix with summed values.
        None: If matrices are not the same shape.
    """
    if len(mat1) != len(mat2):
        return None

    new_matrix = []

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])

        new_matrix.append(row)

    return new_matrix
