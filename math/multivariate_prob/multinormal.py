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

    def pdf(self, x):
        """
        Calculates the PDF of a data point x.

        Parameters
        ----------
        x : numpy.ndarray of shape (d, 1)
            Data point to evaluate.

        Returns
        -------
        float
            The PDF value at x.

        Raises
        ------
        TypeError
            If x is not a numpy.ndarray.
        ValueError
            If x does not have shape (d, 1) or covariance is singular.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        det_cov = np.linalg.det(self.cov)
        if det_cov == 0:
            raise ValueError("Covariance matrix is singular")

        inv_cov = np.linalg.inv(self.cov)
        exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
        pdf_val = (1 / np.sqrt((2 * np.pi)**d * det_cov)) * np.exp(exponent)

        return float(pdf_val)
