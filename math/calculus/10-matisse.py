#!/usr/bin/env python3
"""Task 10: poly_derivative"""


def poly_derivative(poly):
    """Return the derivative of a polynomial as a list"""
    # Check for invalid input
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    # Constant polynomial → derivative is 0
    if len(poly) == 1:
        return [0]

    # Compute derivative
    deriv = [i * poly[i] for i in range(1, len(poly))]

    # If derivative is all zeros, return [0]
    if all(x == 0 for x in deriv):
        return [0]

    return deriv
