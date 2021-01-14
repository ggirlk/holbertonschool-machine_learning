#!/usr/bin/env python3
"""
updates a variable using the gradient
descent with momentum optimization algorithm
"""
import numpy as np


# update_variables_momentum(0.01, 0.9, W, dW, dW_prev)
def update_variables_momentum(alpha, beta1, var, grad, v):
    """ doc """

    vt = np.add(v*beta1, grad*(1-beta1))

    newgrad = np.subtract(var, np.multiply(vt, alpha))
    return newgrad, grad
