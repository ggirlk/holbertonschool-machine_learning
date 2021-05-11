#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain
    being in a particular state after a specified
    number of iterations
    """
    if ((type(P) is not np.ndarray or type(s) is not np.ndarray
         or P.ndim != 2 or s.ndim != 2 or P.shape[0] != P.shape[1]
         or s.shape[0] != 1 or s.shape[1] != P.shape[0]
         or type(t) is not int or t < 0)):
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
