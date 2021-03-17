#!/usr/bin/env python3
""" matrix """


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        if len(matrix) == 1 and len(matrix[0]) == 0:
                return(1)
        raise ValueError("matrix must be a square matrix")
    m = len(matrix)
    det = 0

    def copyMatrix(matrix):
        """ copy list """
        mat = []
        for i in matrix:
            d = []
            for j in i:
                d.append(j)
            mat.append(d)
        return mat

    mat = copyMatrix(matrix)
    for i in range(m):
        for j in range(m):
            del mat[j][i]
        del mat[0]
        det += -(-1)**i*matrix[0][i]*((mat[0][0]*mat[1][1])-(mat[0][1]*mat[1][0]))
        mat = copyMatrix(matrix)

    return det
