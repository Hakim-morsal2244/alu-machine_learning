#!/usr/bin/env python3
"""
Module that defines the MultiNormal class representing
a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal distribution.

        Parameters
        ----------
        data : numpy.ndarray of shape (d, n)
            The data set where n is the number of data points
            and d is the number of dimensions.

        Raises
        ------
        TypeError
            If data is not a 2D numpy.ndarray.
        ValueError
            If n < 2.

        Public instance variables
        -------------------------
        mean : numpy.ndarray of shape (d, 1)
            Mean of the data.
        cov : numpy.ndarray of shape (d, d)
            Covariance matrix of the data.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean (d x 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data
        X_centered = data - self.mean

        # Compute covariance manually (d x d)
        self.cov = np.dot(X_centered, X_centered.T) / (n - 1)
