#!/usr/bin/env python3
""" doc """


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ doc """
    sw = 0
    for i in range(1, L):
        sw += (weights['W'+str(i)]).sum()
    l2 = lambtha*sw/(2*m)
    return (cost+l2)
