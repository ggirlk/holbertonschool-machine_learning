#!/usr/bin/env python3
""" doc """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ doc """
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    imgh = images.shape[1]
    imgw = images.shape[2]
    output = np.zeros((images.shape[0],
                       imgh,
                       imgw))
    for i in range(images.shape[1]):
        if i < images.shape[1] - kh:
            image = images[i]
            for j in range(images.shape[2]):
                if j < images.shape[2] - kw:
                    output[:, i, j] = np.tensordot(images[:,
                                                          i:i+kh,
                                                          j:j+kw],
                                                   kernel)
    return output
