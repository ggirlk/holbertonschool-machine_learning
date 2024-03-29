#!/usr/bin/env python3
""" doc """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ doc """
    m, imgh, imgw, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    imgh, imgw = (imgh-kh)//sh + 1, (imgw-kw)//sw + 1
    output = np.zeros((m, imgh, imgw, c))

    for i in range(imgh):
        for j in range(imgw):
            if mode == 'max':
                output[:, i, j, :] = np.max(A_prev[:,
                                                   i*sh:i*sh+kh,
                                                   j*sw:j*sw+kw, :],
                                            axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.average(A_prev[:,
                                                       i*sh:i*sh+kh,
                                                       j*sw:j*sw+kw, :],
                                                axis=(1, 2))
    return output
