#!/usr/bin/env python3
"""Task 9: summation_i_squared"""


def summation_i_squared(n):
    """Return sum of squares from 1 to n"""
    if type(n) != int or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
