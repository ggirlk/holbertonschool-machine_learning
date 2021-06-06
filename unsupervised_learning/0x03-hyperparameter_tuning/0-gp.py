#!/usr/bin/env python3
""" Hyperparameter Tuning """
import numpy as np


class GaussianProcess():
    """ noiseless 1D Gaussian process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        *************************************************
        ***************** constructor *******************
        *************************************************
        @X_init: is a numpy.ndarray of shape (t, 1)
                 representing the inputs already sampled
                 with the black-box function
        @Y_init: is a numpy.ndarray of shape (t, 1)
                 representing the outputs of the black-box
                 function for each input in X_init
        @t is: the number of initial samples
        @l is: the length parameter for the kernel
        @sigma_f: is the standard deviation given to the
                  output of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel
        matrix between two matrices using
        use the Radial Basis Function (RBF)

        @X1: is a numpy.ndarray of shape (m, 1)
        @X2: is a numpy.ndarray of shape (n, 1)

        Returns: the covariance kernel matrix as
                 a numpy.ndarray of shape (m, n)
        """
        return self.sigma_f**2 * np.exp(pow(X1 - X2.T, 2)/-2/self.l**2)
