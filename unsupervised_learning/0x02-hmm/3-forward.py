#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def forward(Obs, Emiss, Trans, Init):
    """
    performs the forward algorithm
    for a hidden markov model
    """
    T = Obs.shape[0]
    N, M = Emiss.shape
    alpha = np.zeros((N, T))

    for t in range(T):
        for s in range(N):
            if t == 0:
                alpha[s, 0] = Init[s, 0] * Emiss[s, Obs[t]]
            else:
                alpha[s, t] = np.sum(alpha[:, t - 1]
                                     * Trans[:, s]
                                     * Emiss[:, Obs[t]])


    return np.sum(alpha[:, T-1]), alpha 
