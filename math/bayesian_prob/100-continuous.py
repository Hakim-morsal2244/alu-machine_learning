#!/usr/bin/env python3
"""
Module to calculate the continuous posterior probability that the
probability of developing severe side effects falls within a range
given observed data.
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of developing
    severe side effects falls within [p1, p2] given x successes out of n
    trials and assuming a uniform prior.

    Parameters
    ----------
    x : int
        Number of patients that develop severe side effects.
    n : int
        Total number of patients observed.
    p1 : float
        Lower bound of the probability range.
    p2 : float
        Upper bound of the probability range.

    Returns
    -------
    float
        Posterior probability that p is within [p1, p2].

    Raises
    ------
    ValueError
        If n <= 0, x < 0, x > n, p1 or p2 not in [0, 1], or p2 <= p1.
    """
    # Validate n and x
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate p1 and p2
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Compute posterior using Beta CDF
    alpha = x + 1
    beta = n - x + 1

    # CDF values at p2 and p1
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    return cdf_p2 - cdf_p1
