#!/usr/bin/env python3
""" doc """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ doc """
    m, imgh, imgw, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    imgh, imgw = (imgh-kh)//sh + 1, (imgw-kw)//sw + 1
    output = np.zeros((m, imgh, imgw, c))

    for i in range(imgh):
        for j in range(imgw):
            if mode == 'max':
                output[:, i, j, :] = np.max(images[:,
                                                   i*sh:i*sh+kh,
                                                   j*sw:j*sw+kw, :],
                                            axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.average(images[:,
                                                       i*sh:i*sh+kh,
                                                       j*sw:j*sw+kw, :],
                                                axis=(1, 2))
    return output
