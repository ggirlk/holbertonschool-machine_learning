#!/usr/bin/env python3
""" matrix """
import numpy as np


def adjugate(matrix):
    """ calculate the adjugate matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    U,sigma,Vt = np.linalg.svd(matrix)
    N = len(sigma)
    g = np.tile(sigma, N)
    g[::(N+1)] = 1
    G = np.diag(-(-1)**N*np.product(np.reshape(g, (N, N)), 1)) 
    return (U @ G @ Vt).round().transpose().tolist()
