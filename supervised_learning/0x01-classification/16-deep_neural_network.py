#!/usr/bin/env python3
"""
class DeepNeuralNetwork
define a deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork():
    """ Class DeepNeuralNetwork """

    def __init__(self, nx, layers):
        """ Constractor """

        # nx: number of input features to the neuron
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        # layers: number of nodes in each layer of the network
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        def check(b):
            """ check layer list elements """
            if type(b) is not int:
                raise TypeError("layers must be a list of positive integers")
            if b < 1:
                raise ValueError("layers must be a list of positive integers")
        s = list(map(lambda b: check(b), layers))
        # number of layers in the neural network
        self.L = len(layers)
        # intermediary values of the network
        self.cache = {}
        # weights and biased of the network
        self.weights = {}
        for i in range(self.L):
            n = layers[i]
            if i == 0:
                m = self.nx
            else:
                m = layers[i-1]
            self.weights['W' + str(i+1)] = np.random.randn(n, m) * np.sqrt(2/m)
            self.weights['b' + str(i+1)] = np.zeros((n, 1))
