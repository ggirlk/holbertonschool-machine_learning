#!/usr/bin/env python3

def shaperec(matrix, shapelist=[]):
    if type(matrix) == list:
        shapelist.append(len(matrix))
        shaperec(matrix[0], shapelist)


def matrix_shape(matrix):
    shapelist = []
    shaperec(matrix, shapelist)
    return shapelist
