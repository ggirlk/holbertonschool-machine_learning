#!/usr/bin/env python3
""" slice a matrix along a specific axes """


def np_slice(matrix, axes={}):
    """ slice function"""
    axs = list(axes.keys())
    new = []
    for i in axs:
        start = axes[i][0]
        if len(axes[i]) > 1:
            stop = axes[i][1]
        else:
            stop = None
        if len(axes[i]) > 2:
            step = axes[i][2]
        else:
            step = None
        sl = slice (start, stop, step)
        if i == 0:
            matrix = matrix[:start]
        for mat in matrix:
            if (i == 1): 
                new.append(mat[sl])
            if (i == 2):
                new1 = []
                for ma in mat:
                    new1.append(ma[sl])
                new.append(new1)
    return new
