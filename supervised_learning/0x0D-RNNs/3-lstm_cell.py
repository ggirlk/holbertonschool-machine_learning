#!/usr/bin/env python3
""" RNN """
import numpy as np


class LSTMCell():
    """ Long Short Term Memory Cell """

    def __init__(self, i, h, o):
        """
        constructor
        @i: the dimensionality of the data
        @h: the dimensionality of the hidden state
        @o: the dimensionality of the outputs
        """

        self.Wf = np.random.randn(h+i, h)
        self.Wu = np.random.randn(h+i, h)
        self.Wc = np.random.randn(h+i, h)
        self.Wo = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, A):
        """ calculate the softmax """
        e = np.exp(A)
        return e / e.sum(axis=1, keepdims=True)

    def sigmoid(self, z):
        """ sigmoid function """
        return (1/(1+np.exp(-z)))

    def forward(self, h_prev, c_prev, x_t):
        """
        *********************************************
        *** forward propagation for one time step ***
        *********************************************
        @x_t: is a numpy.ndarray of shape (m, i) that
              contains the data input for the cell
                m is the batche size for the data
        @h_prev: is a numpy.ndarray of shape (m, h)
                 containing the previous hidden state
        @c_prev: is a numpy.ndarray of shape (m, h)
                 containing the previous cell state
        Returns: h_next, y
                 h_next is the next hidden state
                 y is the output of the cell
        """
        hx = np.concatenate((h_prev, x_t), 1)

        ai = np.tanh(np.dot(hx, self.Wc) + self.bc)

        f = self.sigmoid(np.dot(hx, self.Wf) + self.bf)
        u = self.sigmoid(np.dot(hx, self.Wu) + self.bu)
        o = self.sigmoid(np.dot(hx, self.Wo) + self.bo)

        ci = f * c_prev + u * ai
        ai = o * np.tanh(ci)
        yi = np.dot(ai, self.Wy) + self.by
        return ai, ci, self.softmax(yi)
