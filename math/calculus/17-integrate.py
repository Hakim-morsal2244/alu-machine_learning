#!/usr/bin/env python3
"""Task 17: poly_integral"""


def poly_integral(poly, C=0):
    """Return the integral of a polynomial as a list"""
    # Validate poly and C
    if (not isinstance(poly, list) or len(poly) == 0
            or not all(isinstance(x, (int, float)) for x in poly)
            or not isinstance(C, (int, float))):
        return None

    # Start with constant C
    integral = [C]

    # Compute integral coefficients
    for i, coef in enumerate(poly):
        val = coef / (i + 1)
        # Convert whole numbers to int
        if val == int(val):
            val = int(val)
        integral.append(val)

    # Trim trailing zeros (keep at least one element)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
