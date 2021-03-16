#!/usr/bin/env python3
""" matrix """
import numpy as np


def inverse(matrix):
    """ calculate the inverse matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    try:
        return np.linalg.inv(matrix).round(10).tolist()
    except Exception:
        return None
