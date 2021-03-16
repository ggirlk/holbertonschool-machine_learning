#!/usr/bin/env python3
""" matrix """
import numpy as np


def minor_of_element(A, i, j):
    """ calculate the minor matrix of one element """
    sub_A = np.delete(A,i-1,0)     # Delete i-th row
    sub_A = np.delete(sub_A,j-1,1) # Delete j-th column
    M_ij = np.linalg.det(sub_A)    # Minor of the element at ith row and jth column
    return np.around(M_ij, decimals=3)  # Rounding the value

def minor(matrix):
    """ calculate the minor matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    m = []
    for i in range(len(matrix)):
        d = []
        for j in range(len(matrix[i])):
            d.append(minor_of_element(matrix, i, j))
        m.append(d)
            
    return m
