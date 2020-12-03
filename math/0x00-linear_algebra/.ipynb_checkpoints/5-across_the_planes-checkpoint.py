#!/usr/bin/env python3
""" add two matrices element-wise """


def add_arrays(arr1, arr2):
    """ add """
    try:
        new = []
        for i in range(0, len(arr1)):
            sm = arr1[i] + arr2[i]
            new.append(sm)
        return(new)
    except Exception:
        return None


def add_matrices2D(mat1, mat2):
    """ add 2D matrices """
    mat = []
    for i in range(0, len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        mat.append(add_arrays(mat1[i], mat2[i]))
    return mat
