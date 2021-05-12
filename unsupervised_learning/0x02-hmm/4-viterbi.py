#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def viterbi(Obs, Emiss, Trans, Init):
    """
    calculates the most likely sequence of
    hidden states for a hidden markov model
    """
    T = Obs.shape[0]
    N, M = Emiss.shape
    V = np.zeros((N, T))
    B = np.zeros((N, T))

    for t in range(T):
        for s in range(N):
            if t == 0:
                V[s, 0] = Init[s, 0] * Emiss[s, Obs[t]]
            else:
                tmp = V[:, t - 1] * Trans[:, s] * Emiss[s, Obs[t]]
                V[s, t] = np.max(tmp)
                B[s, t] = np.argmax(tmp)
    P = np.max(V[:, T-1])
    pointer = np.argmax(V[:, T-1])
    path = [pointer]
    for t in range(T-1, 0, -1):
        pr = int(B[pointer, t])
        path.append(pr)
        pointer = pr

    return path[::-1], P
