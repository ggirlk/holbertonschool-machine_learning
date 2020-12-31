#!/usr/bin/env python3
"""
Class Neuron
Define a single neuron performing binary classification
"""
import numpy as np


class Neuron():
    """ Class Neuron """


    def __init__(self, nx):
        """ nx is number of input features
            must be an integer and >= 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        '''self.__W = np.random.normal(size=(1, nx))'''
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter """
        return self.__W

    @property
    def b(self):
        """ W getter """
        return self.__b

    @property
    def A(self):
        """ W getter """
        return self.__A

    def forward_prop(self, X):
        """ Forward propogation of the neuron """
        aw = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-aw))
        return self.__A

    def cost(self, Y, A):
        """ calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -(1 / m) * loss
        return cost

    def evaluate(self, X, Y):
        """ Evaluates a neurons predictions """
        prop = self.forward_prop(X)
        cost = self.cost(Y, prop)
        x = np.where(prop >= 0.5, 1, 0)
        return x, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = Y.shape[1]
        dz = np.subtract(A, Y)
        dw = np.matmul(dz, X.T) / m
        self.__b = np.subtract(self.__b, alpha * (np.sum(dz) / m))
        self.__W = np.subtract(self.__W, (np.multiply(alpha, dw)))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
