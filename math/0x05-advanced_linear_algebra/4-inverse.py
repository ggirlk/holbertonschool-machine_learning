#!/usr/bin/env python3
""" matrix """


def minorMat(matrix, i, j):
    """ calculates the minor of a matrix """
    c = matrix
    c = c[:i] + c[i+1:]
    for k in range(0, len(c)):
        c[k] = c[k][:j]+c[k][j+1:]
    return c


def determinant(matrix, n=0):
    """ calculates the determinant of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        if len(matrix) == 1 and len(matrix[0]) == 0:
            return(1)
        raise ValueError("matrix must be a non-empty square matrix")
    n = len(matrix)
    for i in range(n):
        if type(matrix[i]) != list:
            raise TypeError("matrix must be a list of lists")
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    det = 0
    for i in range(n):
        m = minorMat(matrix, 0, i)
        det += ((-1)**i)*matrix[0][i]*determinant(m, n-1)
    return det


def minor_of_element(A, i, j):
    """ calculate the minor matrix of one element """
    c = A
    c = c[:i] + c[i+1:]
    for k in range(0, len(c)):
        c[k] = c[k][:j]+c[k][j+1:]
    n = len(c)
    if n == 0:
        return 1
    if n == 1:
        return c[0][0]
    # return (c[0][0]*c[1][1] - c[0][1]*c[1][0])
    return determinant(c)


def minor(matrix):
    """ calculate the minor matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    m = []
    for i in range(len(matrix)):
        if type(matrix[i]) != list:
            raise TypeError("matrix must be a list of lists")
    for i in range(len(matrix)):
        d = []
        for j in range(len(matrix[i])):
            d.append(minor_of_element(matrix, i, j))
        m.append(d)

    return m


def cofactor(matrix):
    """ calculate the minor matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and type(matrix[0]) == list and len(matrix[0]) == 1:
        return([[1]])
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    for i in range(len(matrix)):
        if type(matrix[i]) != list:
            raise TypeError("matrix must be a list of lists")
    m = []
    for i in range(len(matrix)):
        d = []
        for j in range(len(matrix[i])):
            d.append((-1)**(i+j)*minor_of_element(matrix, i, j))
        m.append(d)

    return m


def transpose(matrix):
    """ transpose matrix """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def adjugate(matrix):
    """ calculate the adjugate matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    for i in range(len(matrix)):
        if type(matrix[i]) != list:
            raise TypeError("matrix must be a list of lists")
    return transpose(cofactor(matrix))


def inverse(matrix):
    """ matrix inverse """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for i in range(len(matrix)):
        if type(matrix[i]) != list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")
    try:
        m = matrix[:]
        det = determinant(m)
        # secial case for 2x2 matrix:
        if len(m) == 2:
            return [[m[1][1]/det, -1*m[0][1]/det],
                    [-1*m[1][0]/det, m[0][0]/det]]
        # find matrix of adjugates
        adjugates = adjugate(matrix)
        for r in range(len(adjugates)):
            for c in range(len(adjugates)):
                adjugates[r][c] = adjugates[r][c]/det
        return adjugates
    except Exception:
        return None
