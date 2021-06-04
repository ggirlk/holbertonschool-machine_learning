#!/usr/bin/env python3
""" RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ forward propagation for a simple RNN """
    t, m, i = X.shape
    _, h = h_0.shape
    # numpy.ndarray containing all of the hidden states
    H = np.ndarray((t+1, m, h))
    # numpy.ndarray containing all of the outputs
    Y = np.ndarray((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for k in range(t):
        h_prev = H[k]
        H[k+1], Y[k] = rnn_cell.forward(h_prev, X[k])
    return H, Y
