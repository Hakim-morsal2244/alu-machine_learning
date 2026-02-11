#!/usr/bin/env python3
"""Task 10: poly_derivative"""


def poly_derivative(poly):
    """Return derivative of polynomial as a list"""
    if (not isinstance(poly, list)
            or not all(isinstance(x, (int, float)) for x in poly)):
        return None
    if len(poly) == 1:
        # constant polynomial, derivative is zero
        return [0]
    # compute derivative
    return [i * poly[i] for i in range(1, len(poly))]
