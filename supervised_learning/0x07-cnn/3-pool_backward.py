#!/usr/bin/env python3
""" doc """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ doc """
    m, imgh, imgw, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    imghp, imgwp = 0, 0

    new = A_prev
    newDA = np.zeros_like(A_prev)
    for n in range(m):
        for i in range(imgh):
            for j in range(imgw):
                for k in range(c):
                    if mode == 'max':
                        tmp = new[n,
                                  i*sh:i*sh+kh,
                                  j*sw:j*sw+kw, k]
                        mask = tmp == np.max(tmp)
                        newDA[n,
                              i*sh:i*sh+kh,
                              j*sw:j*sw+kw, k] += np.multiply(dA[n, i, j, k],
                                                              mask)
                    if mode == 'avg':
                        newDA[n,
                              i*sh:i*sh+kh,
                              j*sw:j*sw+kw, k] += dA[n, i, j, k]/kh/kw
    return newDA
