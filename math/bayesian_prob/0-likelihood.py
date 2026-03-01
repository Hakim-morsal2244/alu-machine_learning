#!/usr/bin/env python3
"""
Module for calculating the likelihood of observing data
in a binomial experiment given hypothetical probabilities.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of observing x successes in n trials
    for each hypothetical probability in P.

    Parameters
    ----------
    x : int
        Number of successes observed.
    n : int
        Total number of trials.
    P : numpy.ndarray
        1D array of hypothetical probabilities of success.

    Returns
    -------
    numpy.ndarray
        Likelihood values for each probability in P.

    Raises
    ------
    ValueError
        If n is not positive, x < 0, or x > n, or values in P not in [0,1].
    TypeError
        If P is not a 1D numpy.ndarray.
    """
    # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient
    coef = (
        np.math.factorial(n)
        / (np.math.factorial(x) * np.math.factorial(n - x))
    )

    # Likelihood for each probability (split to multiple lines)
    L = coef * (P ** x) * ((1 - P) ** (n - x))

    return L
