#!/usr/bin/env python3
""" doc """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Calculate one pass of gradient descent on the neuron """
    m = Y.shape[1]

    def dw(dz, x):
        """ weight derivative """
        return np.matmul(dz, x.T)/m

    def db(dz):
        """ bias derivative"""
        return np.sum(dz, axis=1, keepdims=True)/m

    def der(x):
        """ derivatives tanh activation func """
        return 1 - (x**2)

    def dz(wi, dzi, gprimei, i):
        """ z derivative """
        x = np.matmul(wi.T, dzi)
        dgprime = np.multiply(gprimei, cache["D"+str(i)])/keep_prob
        return np.multiply(dgprime, x)

    n = L
    wb = weights.copy()
    dzi = np.subtract(cache['A'+str(n)], Y)
    for i in reversed(range(1, n+1)):
        Ai = cache['A'+str(i)]
        Ai_1 = cache['A'+str(i-1)]
        b = wb['b'+str(i)]
        if i == n:
            dzi = np.subtract(cache['A'+str(n)], Y)
        else:
            w = wb['W'+str(i+1)]
            dzi = dz(w, dzi, der(Ai), i)
        dwi = dw(dzi, Ai_1)
        dbi = db(dzi)
        weights['b'+str(i)] = wb['b'+str(i)]-alpha*dbi
        weights['W'+str(i)] = wb['W'+str(i)]-alpha*dwi
