#!/usr/bin/env python3
""" doc """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ doc """
    m, h, w = images.shape
    kh, kw = kernel.shape
    imghp, imgwp = kh//2, kw//2
    output = np.zeros((m, h, w))
    new = np.pad(images, ((0, 0), (imghp, imghp), (imgwp, imgwp)), 'constant')
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.tensordot(new[:,
                                           i:i+kh,
                                           j:j+kw],
                                           kernel)
    return output
