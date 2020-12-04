#!/usr/bin/env python3
""" slice a matrix along a specific axes """


def np_slice(matrix, axes={}):
    """ slice function"""
    new = []
    for i in range(0, len(matrix.shape)):
        if i not in axes:
            sl = slice(None, None, None)
        else:
            sl = slice(*axes[i])
        new.append(sl)
    return matrix[tuple(new)]
