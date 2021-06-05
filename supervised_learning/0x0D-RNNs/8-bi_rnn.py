#!/usr/bin/env python3
""" RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for a bidirectional RNN"""
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.ndarray((t + 1, m, h * 2))
    H[0, :, :h] = h_0
    H[-1, :, h:] = h_t
    X_rev = np.flip(X, 0)
    for k in range(X.shape[0]):
        hk = H[k]
        H[k + 1, :, :h] = bi_cell.forward(hk[:, :h], X[k])
        hb = H[-k]
        H[-(k + 1), :, h:] = bi_cell.backward(hb[:, h:], X_rev[k])
    return H[1:], bi_cell.output(H[1:])
