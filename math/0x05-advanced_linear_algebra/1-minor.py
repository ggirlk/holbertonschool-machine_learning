#!/usr/bin/env python3
""" matrix """

determinant = __import__('0-determinant').determinant


def minor_of_element(A, i, j):
    """ calculate the minor matrix of one element """
    c = A
    c = c[:i] + c[i+1:]
    for k in range(0, len(c)):
        c[k] = c[k][:j]+c[k][j+1:]
    n = len(c)
    if n == 0:
        return 0
    if n == 1:
        return c[0][0]
    # return (c[0][0]*c[1][1] - c[0][1]*c[1][0])
    return determinant(c)


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
