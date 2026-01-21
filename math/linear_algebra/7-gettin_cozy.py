#!/usr/bin/env python3
"""
Module that contains function to concatenate two 2D matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along axis 0 (rows) or axis 1 (columns).

    Args:
        mat1 (list of list of int/float): First matrix.
        mat2 (list of list of int/float): Second matrix.
        axis (int): 0 to concatenate rows, 1 to concatenate columns.

    Returns:
        list: New matrix with concatenated values.
        None: If matrices cannot be concatenated.
    """
    # Check axis validity
    if axis not in (0, 1):
        return None

    # Concatenate along rows (axis 0)
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    # Concatenate along columns (axis 1)
    if axis == 1:
        if len(mat1) != len(mat2):
            return None

        new_matrix = []
        for i in range(len(mat1)):
            new_matrix.append(mat1[i] + mat2[i])

        return new_matrix
