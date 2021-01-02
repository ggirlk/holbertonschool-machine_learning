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
                raise TypeError("layers must be a list of positive integers")
        s = list(map(lambda b: check(b), layers))
        # number of layers in the neural network
        self.__L = len(layers)
        # intermediary values of the network
        self.__cache = {}
        # weights and biased of the network
        self.__weights = {}
        for i in range(self.L):
            n = layers[i]
            if i == 0:
                m = self.nx
            else:
                m = layers[i-1]
            self.weights['W' + str(i+1)] = np.random.randn(n, m) * np.sqrt(2/m)
            self.weights['b' + str(i+1)] = np.zeros((n, 1))

    @property
    def L(self):
        """ number of layers L getter """
        return self.__L

    @property
    def cache(self):
        """ cache getter """
        return self.__cache

    @property
    def weights(self):
        """ active output getter """
        return self.__weights

    def sigmoid(self, X=None, w=None, b=None, x=None):
        """ sigmoid function """
        if (x is None):
            x = np.matmul(w, X)
            x = np.add(x, b)
        return (1/(1+np.exp(-x)))

    def forward_prop(self, X):
        """
            Calculate the forward
            propagation of the neuron
            using sigmoid activation function
        """
        i = 0
        n = self.L
        self.__cache['A0'] = X
        # active outputs
        for i in range(1, n+1):
            self.__cache['A'+str(i)] = self.sigmoid(self.__cache['A'+str(i-1)],
                                                    self.weights['W'+str(i)],
                                                    self.weights['b'+str(i)]
                                                    )
        return self.cache['A'+str(n)], self.cache

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """

        m = Y.shape[1]
        s = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return -(1 / m) * s

    def evaluate(self, X, Y):
        """ Evaluate the neuron’s predictions """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return (A, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculate one pass of gradient descent on the neuron """
        m = Y.shape[1]

        def dw(dz, x):
            """ weight derivative """
            return np.matmul(dz, x.T)/m

        def db(dz):
            """ bias derivative"""
            return np.mean(dz, axis=1, keepdims=True)

        def der(x):
            """ sigmoid derivative """
            return x * (1-x)

        def dz(wi, dzi, gprimei):
            """ z derivative """
            x = np.matmul(wi.T, dzi)
            return np.multiply(gprimei, x)

        n = self.L
        wb = self.weights.copy()
        dzi = np.subtract(self.cache['A'+str(n)], Y)
        for i in reversed(range(1, n+1)):
            A1 = self.cache['A'+str(i)]
            A0 = self.cache['A'+str(i-1)]
            b = wb['b'+str(i)]
            if i == n:
                dzi = np.subtract(self.cache['A'+str(n)], Y)
            else:
                w = wb['W'+str(i+1)]
                dzi = dz(w, dzi, der(A1))
            dwi = dw(dzi, A0)
            dbi = db(dzi)
            self.__weights['b'+str(i)] = wb['b'+str(i)]-alpha*dbi
            self.__weights['W'+str(i)] = wb['W'+str(i)]-alpha*dwi
