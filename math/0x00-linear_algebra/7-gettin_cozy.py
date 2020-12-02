#!/usr/bin/env python3
""" concatenate 2 matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenate """
    import copy
    if axis == 0:
        mat = mat1 + mat2
        return copy.deepcopy(mat)
    if axis == 1:
        if len(mat1[0]) != len(mat2):
            return None
        mat = mat1[:]
        for i in range(0, len(mat2)):
            for j in mat2[i]:
                mat[i].append(j)
        return copy.deepcopy(mat)
    return None
