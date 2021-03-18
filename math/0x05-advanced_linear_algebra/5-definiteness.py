#!/usr/bin/env python3
""" matrix """
import numpy as np


def is_pos_def(x):
    """ test positive def """
    return np.all(np.linalg.eigvals(x) > 0)


def is_pos_semi_def(x):
    """ test semi positive def """
    return np.all(np.linalg.eigvals(x) >= 0)


def is_neg_def(x):
    """ test negative def """
    return np.all(np.linalg.eigvals(x) < 0)


def is_neg_semi_def(x):
    """ test semi positive def """
    return np.all(np.linalg.eigvals(x) <= 0)


def definiteness(matrix):
    """ calculates the definiteness of a matrix """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix) == 0 or len(matrix) != len(matrix[0]):
        return None

    if is_pos_def(matrix):
        return "Positive definite"

    if is_pos_semi_def(matrix):
        return "Positive semi-definite"

    if is_neg_def(matrix):
        return "Negative definite"

    if is_neg_semi_def(matrix):
        return "Negative semi-definite"

    try:
        np.linalg.cholesky(matrix)
    except Exception:
        return "Indefinite"
    return None
