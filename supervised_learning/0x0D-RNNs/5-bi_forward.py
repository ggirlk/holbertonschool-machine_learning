#!/usr/bin/env python3
""" RNN """
import numpy as np


class BidirectionalCell:
    """Bidirectional cell"""
    def __init__(self, i, h, o):
        """
        constructor
        @i: the dimensionality of the data
        @h: the dimensionality of the hidden state
        @o: the dimensionality of the outputs
        """
        self.Whf = np.random.randn(h+i, h)
        self.Whb = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h*2, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        *********************************************
        *** forward propagation for one time step ***
        *********************************************
        @x_t: is a numpy.ndarray of shape (m, i) that
              contains the data input for the cell
                m is the batche size for the data
        @h_prev: is a numpy.ndarray of shape (m, h)
                 containing the previous hidden state
        Returns: h_next, the next hidden state
        """
        hx = np.concatenate((h_prev, x_t), 1)
        return np.tanh(np.matmul(hx, self.Whf) + self.bhf)
