#!/usr/bin/env python3
""" matrix """
import numpy as np


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        if len(matrix) == 1 and len(matrix[0]) == 0:
                return(1)
        raise ValueError("matrix must be a square matrix")
    return np.linalg.det(np.array(matrix)).round()
