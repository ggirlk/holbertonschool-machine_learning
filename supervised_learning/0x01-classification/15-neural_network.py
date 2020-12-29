#!/usr/bin/env python3
"""
class NeuralNetwork
define a neural network with one hidden layer
performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """ Class NeuralNetwork """

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
        self.__b1 = np.array([[0]] * nodes)
        # activated output of the hidden layer
        self.__A1 = 0

        # weights vector for the neuron
        self.__W2 = np.random.randn(1, nodes)
        # bias for the neuron
        self.__b2 = 0
        # activated output of the neuron (prediction)
        self.__A2 = 0

    # hidden layer
    @property
    def W1(self):
        """ weights getter """
        return self.__W1

    @property
    def b1(self):
        """ biases getter """
        return self.__b1

    @property
    def A1(self):
        """ active output getter """
        return self.__A1

    # output neuron
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

    def dsigm(self, x):
        """ sigmoid derivative """
        # return (np.exp(-x)/(1+np.exp(-x)**2))
        return self.sigmoid(x=x) * (1-self.sigmoid(x=x))

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
        _, self.__A2 = self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        self.__A2 = np.where(self.__A2 >= 0.5, 1, 0)
        return (self.__A2, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculate one pass of gradient descent on the neuron """
        m = X.shape[1]
        n = A1.shape[0]

        def wUpdate(X, A, w, n=1):
            """ update weight """
            x = A - Y
            xlr = (-alpha*n/(m)) * X.T
            return np.add(w, np.matmul(x, xlr))

        # update weights
        # hidden layer
        self.__W1 = wUpdate(X, A1, self.W1)
        # neuron
        self.__W2 = wUpdate(self.A1, A2, self.W2, 1)

        # update biases
        # hidden layer
        ME = np.mean((Y-A1), axis=1)  # Mean Error
        t = []
        for i in range(n):
            # a = (ME[i] * -alpha)
            a = ME[i] * -alpha
            t.append([a])
        t = np.array(t)
        self.__b1 = np.add(self.b1, t)

        # neuron
        ME = np.mean(A2 - Y)  # Mean Error
        self.__b2 += ME * -alpha

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
        if (verbose or graph) is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        # train the model
        costs = []
        k = 0
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            costs.append(self.cost(Y, A2))
            if verbose == True and i-1 == k-1:
                print("Cost after {} iterations: {}".format(i, costs[i]))
                k += step
        # evaluation of the training data after iterations
        self.__A2, cost = self.evaluate(X, Y)
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
        return (self.__A2, cost)
