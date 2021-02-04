#!/usr/bin/env python3
""" doc """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ doc """
    m, h, w = images.shape
    kh, kw = kernel.shape
    imgh, imgw = h - kh + 1, w - kw + 1
    output = np.zeros((m, imgh, imgw))
    for i in range(imgh):
        for j in range(imgw):
            output[:, i, j] = np.tensordot(images[:,
                                                  i:i+kh,
                                                  j:j+kw],
                                                  kernel)
    return output
