#!/usr/bin/env python3
""" slice a matrix along a specific axes """


def np_slice(matrix, axes={}):
    """ slice function"""
    axs = list(axes.keys())
    nmatrix = matrix[:]
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
        if i == 0:
            sl = slice (start)
            nmatrix = nmatrix[sl]
        sl = slice (start, stop, step)
        if (i == 1): 
            nmatrix = nmatrix[:,sl]
        if (i == 2):
            nmatrix = nmatrix[:,:,sl]
    return nmatrix
