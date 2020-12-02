#!/usr/bin/env python3
""" add two matrices element-wise """


def shaperec(matrix, shapelist=[]):
    """ recursion for the shape """
    if type(matrix) == list:
        shapelist.append(len(matrix))
        shaperec(matrix[0], shapelist)


def matrix_shape(matrix):
    """ matrix shape """
    shapelist = []
    shaperec(matrix, shapelist)
    return shapelist


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
    """if (matrix_shape(mat1) != matrix_shape(mat2)):
        return(None)"""
    mat = []
    for i in range(0, len(mat1)):
        mat.append(add_arrays(mat1[i], mat2[i]))
    return mat
