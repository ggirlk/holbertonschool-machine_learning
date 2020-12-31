#!/usr/bin/env python3
"""
class NeuralNetwork
define a neural network with one hidden layer
performing binary classification
"""
import numpy as np


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
        self.__b1 = np.array([[np.array(0.)]] * nodes)
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

        # layer active output
        self.__A1 = self.sigmoid(X, self.W1, self.b1)
        # neuron active output
        self.__A2 = self.sigmoid(self.__A1, self.W2, self.b2)

        return (self.A1, self.A2)

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """

        m = Y.shape[1]
        s = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return -(1 / m) * s

    def evaluate(self, X, Y):
        """ Evaluate the neuron’s predictions """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        A2 = np.where(A2 >= 0.5, 1, 0)
        return (A2, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
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

        def dz(w, dz, gprime):
            """ z derivative """
            x = gprime * np.matmul(w.T, dz)
            return x

        w2 = self.W2
        # output neuron
        dz2 = np.subtract(A2, Y)
        dw2 = dw(dz2, A1)
        self.__W2 = np.subtract(self.W2, np.multiply(alpha, dw2))
        db2 = db(dz2)
        self.__b2 = np.subtract(self.b2, np.multiply(alpha, db2))

        # hidden layer
        dz1 = dz(w2, dz2, der(A1))
        dw1 = dw(dz1, X)
        self.__W1 = np.subtract(self.W1, np.multiply(alpha, dw1))
        db1 = db(dz1)
        self.__b1 = np.subtract(self.b1, np.multiply(alpha, db1))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ train the neuron """
        # check iterations validity
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        # check alpha validity
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        # check step validity
        if (verbose or graph) is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        # train the model
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        # return the evaluation of the training data after iterations
        return self.evaluate(X, Y)
