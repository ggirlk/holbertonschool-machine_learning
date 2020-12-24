#!/usr/bin/env python3
"""
Class Neuron
Define a single neuron performing binary classification
"""
import numpy as np


class Neuron():
    """ Class Neuron"""

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
