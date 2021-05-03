#!/usr/bin/env python3

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ doc """
    try:
        pi, m, S = initialize(X, k)
        i = 0
        prevl = 0
        g, l = expectation(X, pi, m, S)
        while i < iterations:
            pi, m, S = maximization(X, g)
            if verbose and (not i%10 or round(prevl, 5) == round(l, 5)):
                print("Log Likelihood after {} iterations: {}".format(i, round(l, 5)))
            if l >= tol or round(prevl, 5) == round(l, 5):
                return pi, m, S, g, l
            prevl = l
            g, l = expectation(X, pi, m, S)
            i += 1
            
        return pi, m, S, g, l
    except Exception:
        return None, None, None, None, None
    