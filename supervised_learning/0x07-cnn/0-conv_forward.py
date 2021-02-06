#!/usr/bin/env python3
""" doc """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ doc """
    m, imgh, imgw, c = A_prev.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    imghp, imgwp = 0, 0
    if padding == 'same':
        imghp = (((imgh - 1) * sh + kh - imgh) // 2) + 1
        imgwp = (((imgw - 1) * sw + kw - imgw) // 2) + 1

    if type(padding) == tuple:
        imghp, imgwp = padding
    imgh, imgw = (imgh-kh+2*imghp)//sh + 1, (imgw-kw+2*imgwp)//sw + 1
    output = np.zeros((m, imgh, imgw, c))
    new = np.pad(A_prev, ((0, 0), (imghp, imghp),
                          (imgwp, imgwp), (0, 0)),
                 'constant')
    for k in range(knc):
        for i in range(imgh):
            for j in range(imgw):
                output[:, i, j, k] = np.sum(new[:,
                                                i*sh:i*sh+kh,
                                                j*sw:j*sw+kw, :]
                                            * W[..., k],
                                            axis=(1, 2, 3))
    return activation(output)