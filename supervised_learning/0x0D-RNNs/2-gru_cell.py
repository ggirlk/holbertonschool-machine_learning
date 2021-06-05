#!/usr/bin/env python3
""" RNN """
import numpy as np


class GRUCell():
    """ Gated Recurrent Unit Cell """

    def __init__(self, i, h, o):
        """
        constructor
        @i: the dimensionality of the data
        @h: the dimensionality of the hidden state
        @o: the dimensionality of the outputs
        """

        self.Wh = np.random.randn(h+i, h)
        self.Wz = np.random.randn(h+i, h)
        self.Wr = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.by = np.zeros((1, o))


    def softmax(self, A):
        """ calculate the softmax """
        e = np.exp(A)
        return e / e.sum(axis=1, keepdims=True)

    def sigmoid(self, z):
        """ sigmoid function """
        return (1/(1+np.exp(-z)))

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
        Returns: h_next, y
                 h_next is the next hidden state
                 y is the output of the cell
        """
        hx = np.concatenate((h_prev, x_t), 1)

        g1 = self.sigmoid(np.dot(hx, self.Wr) + self.br)
        g2 = self.sigmoid(np.dot(hx, self.Wz) + self.bz)

        hx = np.concatenate((g1 * h_prev, x_t), 1)
        ai = np.tanh(np.dot(hx, self.Wh) + self.bh)
        ai = (1 - g2) * h_prev + g2 * ai
        yi = np.dot(ai, self.Wy) + self.by
        return ai, self.softmax(yi)

