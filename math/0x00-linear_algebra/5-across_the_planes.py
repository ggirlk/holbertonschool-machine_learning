#!/usr/bin/env python3
def shaperec(matrix, shapelist=[]):
    if type(matrix) == list:
        shapelist.append(len(matrix))
        shaperec(matrix[0], shapelist)


def matrix_shape(matrix):
    shapelist = []
    shaperec(matrix, shapelist)
    return shapelist


def add_arrays(arr1, arr2):
    if (matrix_shape(arr1) != matrix_shape(arr2)):
        return(None)
    new = []
    for i in range(0, len(arr1)):
        sm = arr1[i] + arr2[i]
        new.append(sm)
    return(new)


def add_matrices2D(mat1, mat2):
    if (matrix_shape(mat1) != matrix_shape(mat2)):
        return(None)
    mat = []
    for i in range(0, len(mat1)):
        mat.append(add_arrays(mat1[i], mat2[i]))
    return mat
