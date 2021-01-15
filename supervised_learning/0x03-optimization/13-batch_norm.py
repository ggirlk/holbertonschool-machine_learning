#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ doc """
    µ = Z.mean(axis=0)
    var = (np.subtract(Z, µ)**2).mean(axis=0)
    znorm = np.subtract(Z, µ)/(np.sqrt(var+epsilon))
    return gamma*znorm + beta
