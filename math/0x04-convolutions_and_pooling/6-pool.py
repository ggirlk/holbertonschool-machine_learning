#!/usr/bin/env python3
""" doc """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ doc """
    m, imgh, imgw, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    imghp, imgwp = 0, 0
    # imghp = (((imgh - 1) * sh + kh - imgh) // 2) + 1
    # imgwp = (((imgw - 1) * sw + kw - imgw) // 2) + 1
    imgh, imgw = (imgh-kh+2*imghp)//sh + 1, (imgw-kw+2*imgwp)//sw + 1
    output = np.zeros((m, imgh, imgw))
    new = np.pad(images, ((0, 0), (imghp, imghp),
                          (imgwp, imgwp), (0, 0)),
                 'constant')

    for i in range(imgh):
        for j in range(imgw):
            if mode == 'max':
                output[:, i, j] = np.max(new[:,
                                                i*sh:i*sh+kh,
                                                j*sw:j*sw+kw, :],
                                            axis=(1, 2))
            if mode == 'avg':
                output[:, i, j] = np.average(new[:,
                                                    i*sh:i*sh+kh,
                                                    j*sw:j*sw+kw, :],
                                                axis=(1, 2))
    return output
