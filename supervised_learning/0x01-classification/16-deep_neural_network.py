#!/usr/bin/env python3
"""
class DeepNeuralNetwork
define a deep neural network performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


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
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        for i in layers:
            if type(i) is not int or i < 0:
                raise ValueError("layers must be a list of positive integers")
        # number of layers in the neural network
        self.L = len(layers)
        # intermediary values of the network
        self.cache = {}
        # weights and biased of the network
        self.weights = {}
        for i in range(self.L):
            n = layers[i]
            m = layers[i-1]
            if i == 0:
                m = self.nx
            self.weights['W' + str(i+1)] = np.random.randn(n, m)
            self.weights['b' + str(i+1)] = np.zeros((n, 1))