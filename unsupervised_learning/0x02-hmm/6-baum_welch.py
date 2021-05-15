#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def baum_welch(Obs, Trans, Emiss, Init, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model
    """
    T = Obs.shape[0]
    N, M = Emiss.shape

    def forward(Obs, Trans, Emiss, Init):
        """ baum welch forward """
        alpha = np.ndarray((N, T))
        state = Init[:, 0]
        for t in range(T):
            state = state * Emiss[:, Obs[t]]
            alpha[:, t] = state
            state = np.matmul(Trans.T, state)
        return alpha[:, -1].sum(), alpha

    def backward(Obs, Trans, Emiss, Init):
        """ baum welch backward """
        beta = np.ndarray((N, T))
        state = np.asarray([1] * N)
        beta[:, -1] = state
        for t in range(T - 2, -1, -1):
            state = np.matmul(Trans, state * Emiss[:, Obs[t + 1]])
            beta[:, t] = state
        return (beta[:, 0] * Init[:, 0]
                * Emiss[:, Obs[0]]).sum(), beta
    transprev = None
    while iterations:
        iterations -= 1
        P, alpha = forward(Obs, Trans, Emiss, Init)
        P2, beta = backward(Obs, Trans, Emiss, Init)

        forbackxi = alpha[:, None, :-1] * beta[None, :, 1:]
        emittedprobs = Emiss[:, Obs[1:]]
        xi = forbackxi * Trans[..., None] * emittedprobs[None, :, ...]
        xi /= xi.sum(axis=(0, 1))
        forbackga = alpha * beta
        gamma = forbackga / forbackga.sum(axis=0)
        Trans = xi.sum(axis=2) / xi.sum(axis=(1, 2))
        Trans = Trans.T
        for emit in range(M):
            gammanum = gamma[:, Obs == emit]
            Emiss[:, emit] = gammanum.sum(axis=1) / gamma.sum(axis=1)
        if ((np.all(transprev == Trans)
             and np.all(emprev == Emiss)
             and np.all(initprev == Init))):
            return Trans, Emiss
        transprev = Trans
        initprev = Init
        emprev = Emiss
    return Trans, Emiss
