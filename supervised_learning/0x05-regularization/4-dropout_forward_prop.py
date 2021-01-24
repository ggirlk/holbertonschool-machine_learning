#!/usr/bin/env python3
""" doc """
import tensorflow as tf
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ doc """
    def softmax(z):
        """ softmax function """
        return np.exp(z)/(np.sum(np.exp(z), axis=0))
    def tanh(z):
        """ tanh function """
        return np.tanh(z)
    np.random.seed(1)
    cache = {}
    cache["A0"] = X
    for i in range(1, L+1):
        W = weights["W"+str(i)]
        b = weights["b"+str(i)]
        A = cache['A'+str(i-1)]
        Z = np.dot(W, A) + b
        if i != L:
            cache['A'+str(i)] = tanh(Z)
            A = cache['A'+str(i)]
            cache["D"+str(i)] = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            cache["D"+str(i)] = np.where(cache["D"+str(i)] < keep_prob, 0, 1)
            cache['A'+str(i)] = np.multiply(A, cache["D"+str(i)]) / keep_prob
        else:
            cache['A'+str(i)] = softmax(Z)
    return cache
