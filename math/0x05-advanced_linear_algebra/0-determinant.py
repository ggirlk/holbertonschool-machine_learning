#!/usr/bin/env python3
""" matrix """


def minorMat(matrix, i, j):
    """ calculates the minor of a matrix """
    c = matrix
    c = c[:i] + c[i+1:]
    for k in range(0, len(c)):
        c[k] = c[k][:j]+c[k][j+1:]
    return c


def determinant(matrix, n=0):
    """ calculates the determinant of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        if len(matrix) == 1 and len(matrix[0]) == 0:
            return(1)
        raise ValueError("matrix must be a square matrix")
    n = len(matrix)
    for i in range(n):
        if type(matrix[i]) != list:
            raise TypeError("matrix must be a list of lists")
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    det = 0
    for i in range(n):
        m = minorMat(matrix, 0, i)
        det += ((-1)**i)*matrix[0][i]*determinant(m, n-1)
    return det
