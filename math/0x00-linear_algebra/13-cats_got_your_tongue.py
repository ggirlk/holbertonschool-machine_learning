#!/usr/bin/env python3
""" concatenates two matrices along a specific axis """


def np_cat(mat1, mat2, axis=0):
    """ concatenates two matrices """
    import numpy as np
    return np.concatenate((mat1, mat2), axis=axis)
