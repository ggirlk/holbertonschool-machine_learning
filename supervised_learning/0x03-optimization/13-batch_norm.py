#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""


def batch_norm(Z, gamma, beta, epsilon):
    """ doc """
    µ = Z.mean()
    var = ((Z-µ)**2).mean()
    znorm = (Z-µ)/(np.sqrt(var+epsilon))
    return gamma*znorm + beta
