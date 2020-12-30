#!/usr/bin/env python3
"""
class NeuralNetwork
define a neural network with one hidden layer
performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """ Class Neuron """

    def __init__(self, nx, nodes):
        """ Constractor """
        
        # nx: number of input features to the neuron
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        # nodes: number of nodes found in the hidden layer
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # weights vector for the hidden layer
        self.__W1 = np.random.randn(nodes, self.nx)
        # bias for the hidden layer
        self.__b1 = np.array([[0]]* nodes)
        # activated output of the hidden layer
        self.__A1 = 0

        # weights vector for the neuron
        self.__W2 = np.random.randn(1, nodes)
        # bias for the neuron
        self.__b2 = 0
        # activated output of the neuron (prediction)
        self.__A2 = 0

    # the hidden layer
    @property
    def W1(self):
        """ weights getter """
        return self.__W1

    @property
    def b1(self):
        """ bias getter """
        return self.__b1

    @property
    def A1(self):
        """ active output getter """
        return self.__A1

    # the neuron
    @property
    def W2(self):
        """ weights getter """
        return self.__W2

    @property
    def b2(self):
        """ bias getter """
        return self.__b2

    @property
    def A2(self):
        """ active output getter """
        return self.__A2
