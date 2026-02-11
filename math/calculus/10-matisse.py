#!/usr/bin/env python3
"""Task 10: poly_derivative"""

def poly_derivative(poly):
    """Return derivative of polynomial as list"""
    if type(poly) != list or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if len(poly) <= 1:
        return [0]
    return [i * poly[i] for i in range(1, len(poly))]
