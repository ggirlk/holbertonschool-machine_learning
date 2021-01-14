#!/usr/bin/env python3
"""
updates a variable using the gradient
descent with momentum optimization algorithm
"""
import numpy as np


# update_variables_momentum(0.01, 0.9, W, dW, dW_prev)
def update_variables_momentum(alpha, beta1, var, grad, v):
    """ doc """
    m_avg = []
    vt=0
    for i in range(len(grad)):
        vt = (vt*beta1 + grad[i]*(1-beta1))
        # avg = vt/(1-beta1**(i+1))
        m_avg.append(vt)

    newgrad = np.array(m_avg)
    newgrad = np.subtract(var, np.multiply(newgrad, alpha))
    return newgrad, grad
