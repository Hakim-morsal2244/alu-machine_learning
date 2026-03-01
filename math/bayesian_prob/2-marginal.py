#!/usr/bin/env python3
"""
Module to calculate the marginal probability of observed data
given hypothetical probabilities and prior beliefs.
"""
import numpy as np
import importlib

# Dynamically import intersection function
intersection = importlib.import_module("1-intersection").intersection


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining x successes
    out of n trials given hypothetical probabilities P and prior Pr.

    Parameters
    ----------
    x : int
        Number of successes observed.
    n : int
        Total number of trials.
    P : numpy.ndarray
        1D array of hypothetical probabilities of success.
    Pr : numpy.ndarray
        1D array of prior beliefs for each probability in P.

    Returns
    -------
    float
        The marginal probability of obtaining x and n.

    Raises
    ------
    ValueError
        If n is not positive, x < 0, x > n, or Pr does not sum to 1.
    TypeError
        If P is not 1D numpy.ndarray, or Pr is not same shape as P.
    ValueError
        If any value in P or Pr is not in [0, 1].
    """
    # Validation (in order)
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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Marginal probability = sum of intersections
    I = intersection(x, n, P, Pr)
    return np.sum(I)
