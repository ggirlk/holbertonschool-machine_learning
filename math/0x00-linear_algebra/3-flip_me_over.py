#!/usr/bin/env python3
def matrix_transpose(matrix):
    mtrans = []
    for i in range(0, len(matrix[0])):
        mat = []
        for m in matrix:
            mat.append(m[i])
        mtrans.append(mat)
    return(mtrans)
