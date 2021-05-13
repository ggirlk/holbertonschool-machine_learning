#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def backward(Obs, Emiss, Trans, Init):
    """
    performs the backward algorithm
    for a hidden markov model
    """
    T = Obs.shape[0]
    N, M = Emiss.shape
    alpha = np.zeros((N, T))
    state = np.asarray([1] * N)
    alpha[:, -1] = state
    for t in range(T - 2, -1, -1):
        state = np.matmul(Trans, state * Emiss[:, Obs[t + 1]])
        alpha[:, t] = state
    return (alpha[:, 0] * Init[:, 0]
            * Emiss[:, Obs[0]]).sum(), alpha
