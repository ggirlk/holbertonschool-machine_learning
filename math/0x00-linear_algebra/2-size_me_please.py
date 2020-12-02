#!/usr/bin/env python3
""" calculates the shape of a matrix """


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
