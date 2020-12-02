#!/usr/bin/env python3
""" matrix multiplication """


def mat_mul(mat1, mat2):
    """ matrix multiplication """
    mat = []
    for i in range(0, len(mat1)):
        res = []
        for j in range(0, len(mat2[0])):
            sm = 0
            for k in range(0, len(mat2)):
                sm += mat1[i][k] * mat2[k][j]
            res.append(sm)
        mat.append(res)
    return(mat)
