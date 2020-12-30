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

    def evaluate(self, X, Y):
        """ Evaluate the neuron’s predictions """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        self.__A = np.where(self.forward_prop(X) >= 0.5, 1, 0)
        return (self.__A, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculate one pass of gradient descent on the neuron """
        m = X.shape[1]

        def dw(dz, x):
            """ weight derivative """
            return np.matmul(dz, x.T)/m

        def db(dz):
            """ bias derivative"""
            return np.mean(dz)

        dz = A - Y
        # update weights
        dw = dw(dz, X)
        self.__W += -alpha * dw
        # update bias
        db = db(dz)
        self.__b += db * -alpha

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
        if verbose or graph:
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
            if verbose == True and i-1 == k-1:
                print("Cost after {} iterations: {}".format(i, costs[i]))
                k += step
        # evaluation of the training data after iterations
        self.__A, cost = self.evaluate(X, Y)
        # last iteration
        i += 1
        print("Cost after {} iterations: {}".format(i, cost))
        # ploting
        if graph == True:
            plt.plot(costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        # return the evaluation
        return (self.__A, cost)
