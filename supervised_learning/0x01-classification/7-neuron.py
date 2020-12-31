#!/usr/bin/env python3
"""
Class Neuron
Define a single neuron performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """ Constractor """

        """nx: number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        """weights vector for the neuron"""
        self.__W = np.random.randn(1, nx)
        """bias for the neuron"""
        self.__b = 0
        """activated output of the neuron (prediction)"""
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

        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1+np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """

        return np.mean(-1 * (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """ Evaluate the neuron’s predictions """
        self.__A = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        A = np.where(self.__A >= 0.5, 1, 0)
        return A, cost

    def dw(self, dz, X, m):
        """ weight derivative """
        return np.matmul(dz, X.T)/m

    def db(self, dz, m):
        """ bias derivative"""
        return np.sum(dz)/m

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculate one pass of gradient descent on the neuron """
        m = Y.shape[1]
        dz = np.subtract(A, Y)
        # update weights
        dw = self.dw(dz, X, m)
        self.__W = np.subtract(self.__W, np.multiply(alpha, dw))
        # update bias
        db = self.db(dz, m)
        self.__b = np.subtract(self.__b, np.multiply(alpha, db))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
        check = 0
        if verbose:
            check = 1
        if graph:
            check = 1
        if check == 1:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        # train the model
        costs = []
        k = 0
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            costs.append(self.cost(Y, A))
            if verbose and i-1 == k-1:
                print("Cost after {} iterations: {}".format(i, costs[i]))
                k += step
        # evaluation of the training data after iterations
        self.__A, cost = self.evaluate(X, Y)
        # last iteration
        i += 1
        print("Cost after {} iterations: {}".format(i, cost))
        # ploting
        if graph:
            plt.plot(costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        # return the evaluation
        return (self.__A, cost)
