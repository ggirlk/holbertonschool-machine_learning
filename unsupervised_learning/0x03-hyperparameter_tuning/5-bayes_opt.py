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

        Z = np.zeros(sigs.shape[0])

        for i in range(sigs.shape[0]):
            if sigs[i] != 0:
                Z[i] = improves[i] / sigs[i]
            else:
                Z[i] = 0
        eis = improves * norm.cdf(Z) + sigs * norm.pdf(Z)
        return self.X_s[np.argmax(eis)], eis

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        @iterations:is the maximum number of iterations to perform

        *** If the next proposed point is one that has already been sampled,
            optimization should be stopped early
        Returns: X_opt, Y_opt
                 X_opt: is a numpy.ndarray of shape (1,) representing
                        the optimal point
                 Y_opt: is a numpy.ndarray of shape (1,) representing
                        the optimal function value
        """

        X_all_s = []
        while iterations:
            iterations -= 1
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)

            x_opt, _ = self.acquisition()
            # If the next proposed point is one that has already been sampled,
            # optimization should be stopped early
            if x_opt in X_all_s:
                break

            y_opt = self.f(x_opt)

            # Add the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            indx = np.argmin(self.gp.Y)
        else:
            indx = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[indx]
        y_opt = self.gp.Y[indx]

        return x_opt, y_opt
