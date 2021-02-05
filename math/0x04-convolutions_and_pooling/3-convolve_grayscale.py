#!/usr/bin/env python3
""" doc """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ doc """
    m, imgh, imgw = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        imghp, imgwp = kh//2, kw//2
    if padding == 'valid':
        imghp, imgwp = 0, 0
    if type(padding) == tuple:
        imghp, imgwp = padding
    imgh, imgw = (imgh-kh+2*imghp)//sh + 1, (imgw-kw+2*imgwp)//sw + 1
    output = np.zeros((m, imgh, imgw))
    new = np.pad(images, ((0, 0), (imghp, imghp), (imgwp, imgwp)), 'constant')
    for i in range(imgh):
        for j in range(imgw):
            output[:, i, j] = np.tensordot(new[:,
                                           i*sh:i*sh+kh,
                                           j*sw:j*sw+kw],
                                           kernel)
    return output
