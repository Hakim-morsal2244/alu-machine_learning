#!/usr/bin/env python3


"""
3-adjugate.py
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        submatrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)

    return det


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.
    """
    n = len(matrix)
    cof = []

    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [
                r[:j] + r[j + 1:]
                for k, r in enumerate(matrix) if k != i
            ]
            row.append(((-1) ** (i + j)) * determinant(submatrix))
        cof.append(row)

    return cof


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    cof = cofactor(matrix)
    n = len(cof)

    # transpose cofactor matrix
    adj = [[cof[j][i] for j in range(n)] for i in range(n)]

    return adj
