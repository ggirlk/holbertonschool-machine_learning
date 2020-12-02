#!/usr/bin/env python3
""" add two arrays element-wise """


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
    if (matrix_shape(arr1) != matrix_shape(arr2)):
        return(None)
    new = []
    for i in range(0, len(arr1)):
        sm = arr1[i] + arr2[i]
        new.append(sm)
    return(new)
