#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    # Check if number of rows are the same
    if len(mat1) != len(mat2):
        return None

    new_matrix = []

    # Add element-wise
    for i in range(len(mat1)):
        # Check if number of columns are the same
        if len(mat1[i]) != len(mat2[i]):
            return None

        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])

        new_matrix.append(row)

    return new_matrix
