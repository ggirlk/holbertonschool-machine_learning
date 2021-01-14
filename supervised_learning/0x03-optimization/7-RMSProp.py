#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ doc """
    st = np.add(s*beta2, (grad**2)*(1-beta2))

    newgrad = np.subtract(var,
                          np.divide(np.multiply(grad, alpha),
                                    (np.sqrt(st)+epsilon)))

    return newgrad, st
