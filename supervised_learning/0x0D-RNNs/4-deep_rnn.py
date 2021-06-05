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


def deep_rnn(rnn_cells, X, h_0):
    """
    *********************************************
    ***** forward propagation for a deep RNN ****
    *********************************************
    @rnn_cells: is a list of RNNCell instances
                of length l that will be used for
                the forward propagation
                 l: is the number of layers
    @X: data to be used, given as a numpy.ndarray
        of shape (t, m, i)
             t: is the maximum number of time steps
             m: is the batch size
             i: is the dimensionality of the data
    @h_0: is the initial hidden state, given as
          a numpy.ndarray of shape (l, m, h)
             h: dimensionality of the hidden state
    Returns:
         H: numpy.ndarray containing all of the hidden states
         Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    # numpy.ndarray containing all of the hidden states
    H = np.ndarray((t + 1, len(rnn_cells), m, h))
    H[0] = h_0
    H[:, 0], _ = rnn(rnn_cells[0], X, h_0[0])
    for k in range(1, len(rnn_cells)):
        H[:, k], Y = rnn(rnn_cells[k], H[1:, k - 1], h_0[k])
    return H, Y
