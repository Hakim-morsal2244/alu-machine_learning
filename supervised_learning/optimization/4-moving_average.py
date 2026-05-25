#!/usr/bin/env python3
"""Moving average with bias correction"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates weighted moving average with bias correction

    data: list of values
    beta: weight factor

    Returns: list of moving averages
    """

    v = 0
    corrected = []
    moving_avgs = []

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x

        # bias correction
        v_corrected = v / (1 - beta ** t)

        moving_avgs.append(v_corrected)

    return moving_avgs
