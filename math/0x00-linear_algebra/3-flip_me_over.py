#!/usr/bin/env python3
""" returns the transpose of a 2D matrix  """


def matrix_transpose(matrix):
    """ matrix transpose """
    mtrans = []
    for i in range(0, len(matrix[0])):
        mat = []
        for m in matrix:
            mat.append(m[i])
        mtrans.append(mat)
    return(mtrans)
