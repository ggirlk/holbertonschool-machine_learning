#!/usr/bin/env python3
""" Hyperparameter Tuning """
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ Initialize Bayesian Optimization """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        *************************************************
        ***************** constructor *******************
        *************************************************
        @f: is the black-box function to be optimized
        @X_init: is a numpy.ndarray of shape (t, 1) representing 
                 the inputs already sampled with the black-box function
        @Y_init: is a numpy.ndarray of shape (t, 1) representing 
                 the outputs of the black-box function for each input in X_init
        @t: is the number of initial samples
        @bounds: is a tuple of (min, max) representing the bounds of
                 the space in which to look for the optimal point
        @ac_samples: is the number of samples that should be analyzed
                 during acquisition
        @l: is the length parameter for the kernel
        @sigma_f: is the standard deviation given to the output of
                  the black-box function
        @xsi: is the exploration-exploitation factor for acquisition
        @minimize: is a bool determining whether optimization should be
                  performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)[:, None]
        self.xsi = xsi
        self.minimize = minimize
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

    def acquisition(self):
        """
        calculates the next best sample location by
        Using the Expected Improvement acquisition function
        Returns: X_next, EI
                 X_next: is a numpy.ndarray of shape (1,)
                 representing the next best sample point
                 EI: is a numpy.ndarray of shape (ac_samples,)
                 containing the expected improvement of each potential sample
        """
        mu, sig = self.gp.predict(self.gp.X)
        next_mu, sigs = self.gp.predict(self.X_s)
        opt = np.min(mu)
        improves = opt - next_mu - self.xsi
        if not self.minimize:
            improve = -improves

        Z = improves/sigs
        eis = improves * norm.cdf(Z) + sigs * norm.pdf(Z)
        return self.X_s[np.argmax(eis)], eis



    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        @iterations is the maximum number of iterations to perform

        *** If the next proposed point is one that has already been sampled,
            optimization should be stopped early
        Returns: X_opt, Y_opt
                 X_opt: is a numpy.ndarray of shape (1,) representing
                        the optimal point
                 Y_opt: is a numpy.ndarray of shape (1,) representing
                        the optimal function value
        """
        prev = None
        f_x = None
        f_y = None
        while iterations:
            X_next, eis = self.acquisition()
            new_y = self.f(X_next)
            if X_next == prev:
                break
            self.gp.update(X_next, new_y)
            pycodehack = f_y is None or self.minimize and f_y > new_y
            if ((pycodehack or not self.minimize and f_y < new_y)):
                f_y = new_y
                f_x = X_next
            prev = X_next
            iterations -= 1
        return f_x, f_y
