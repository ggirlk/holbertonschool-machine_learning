#!/usr/bin/env python3
""" doc """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ doc """
    sw = 0
    for i in range(1, L+1):
        sw += np.linalg.norm(weights['W'+str(i)],
                             ord='fro')**2
    l2 = (lambtha*sw)/(2*m)
    return (cost+l2)
