#!/usr/bin/env python3

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ doc """
    if type(X) is not np.ndarray or X.ndim != 2\
       or type(verbose) is not bool or tol < 0\
       or type(k) is not int or k < 0:
        return None, None, None, None, None
    try:
        pi, m, S = initialize(X, k)
        i = 0
        prevl = 0
        while i < iterations:
            g, l = expectation(X, pi, m, S)
            if verbose and not i%10:
                print("Log Likelihood after {} iterations: {}".format(i, round(l, 5)))
            if abs(prevl - l) < tol:
                print("Log Likelihood after {} iterations: {}".format(i, round(l, 5)))
                return pi, m, S, g, l
            prevl = l
            pi, m, S = maximization(X, g)
            i += 1

        return pi, m, S, g, l
    except Exception:
        return None, None, None, None, None
    