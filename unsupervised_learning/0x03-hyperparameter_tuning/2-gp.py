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

    def predict(self, X_s):
        """
        predicts the mean and standard deviation
        of points in a Gaussian process

        @X_s: is a numpy.ndarray of shape (s, 1) containing all of
              the points whose mean and standard deviation should
              be calculated
        @s: is the number of sample points
        Returns: mu, sigma
                 mu: is a numpy.ndarray of shape (s,) containing the mean
                     for each point in X_s, respectively
                 sigma: is a numpy.ndarray of shape (s,) containing
                        the variance for each point in X_s, respectively
        """
        K_s = self.kernel(X_s, self.X)
        K_inv = np.linalg.inv(self.K)
        mu = np.matmul(np.matmul(K_s, K_inv), self.Y)[:, 0]
        K_s2 = self.kernel(X_s, X_s)
        sigma = K_s2 - np.matmul(np.matmul(K_s, K_inv), K_s.T)
        return mu, np.diagonal(sigma)

    def update(self, X_new, Y_new):
        """
        updates a Gaussian Process
        => Updates the public instance attributes X, Y, and K

        X_new: is a numpy.ndarray of shape (1,)
               that represents the new sample point
        Y_new: is a numpy.ndarray of shape (1,)
               that represents the new sample function value
        """
        self.X = np.append(self.X, X_new[:, None], axis=0)
        self.Y = np.append(self.Y, Y_new[:, None], axis=0)
        self.K = self.kernel(self.X, self.X)