#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2,
                          epsilon, var, grad, v, s, t):
    """ doc """
    st = np.add(s*beta2, (grad**2)*(1-beta2))
    sc = st / (1-beta2**t)

    vt = np.add(v*beta1, grad*(1-beta1))
    vc = vt / (1-beta1**t)

    newgrad = np.subtract(var,
                          np.divide(np.multiply(vc, alpha),
                                    (np.sqrt(sc)+epsilon)))

    return newgrad, vt, st
