#!/usr/bin/env python3
""" doc """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ doc """
    m, imgh, imgw = images.shape
    kh, kw = kernel.shape
    if padding == 'same':
        imghp, imgwp = kh//2, kw//2
    if padding == 'valid':
        imghp, imgwp = 0, 0
    if type(padding) == tuple:
        imghp, imgwp = padding
    imgh, imgw = imgh - kh + 2*imghp + 1, imgw - kw + 2*imgwp + 1
    output = np.zeros((m, imgh, imgw))
    new = np.pad(images, ((0, 0), (imghp, imghp), (imgwp, imgwp)), 'constant')
    for i in range(imgh):
        for j in range(imgw):
            output[:, i, j] = np.tensordot(new[:,
                                           i:i+kh,
                                           j:j+kw],
                                           kernel)
    return output
