#!/usr/bin/env python3
"""
Class Neuron
Define a single neuron performing binary classification
"""
import numpy as np


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """ Constractor """

        # nx: number of input features to the neuron
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        # weights vector for the neuron
        self.__W = np.random.randn(1, self.nx)
        # bias for the neuron
        self.__b = 0
        # activated output of the neuron (prediction)
        self.__A = 0

    @property
    def W(self):
        """ weights getter """
        return self.__W

    @property
    def b(self):
        """ bias getter """
        return self.__b

    @property
    def A(self):
        """ active output getter """
        return self.__A

    def forward_prop(self, X):
        """
            Calculate the forward
            propagation of the neuron
            using sigmoid activation function
        """

        x = np.matmul(self.__W, X) + self.b
        # sigmoid
        self.__A = (1/(1+np.exp(-x)))
        return self.__A

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """

        m = A.shape[1]
        s = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return -(1 / m) * s
