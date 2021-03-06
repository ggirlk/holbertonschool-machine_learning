#!/usr/bin/env python3
""" doc """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ doc """
    m, imgh, imgw, c = dZ.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    imghp, imgwp = 0, 0
    if padding == 'same':
        imghp = (((imgh * sh) - sh + kh - imgh) // 2) + 1
        imgwp = (((imgw * sw) - sw + kw - imgw) // 2) + 1
    if type(padding) == tuple:
        imghp, imgwp = padding

    new = np.pad(A_prev, ((0, 0), (imghp, imghp),
                          (imgwp, imgwp), (0, 0)),
                 'constant', constant_values=0)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    newDZ = np.zeros(new.shape)
    dW = np.zeros_like(W)
    for n in range(m):
        for i in range(imgh):
            for j in range(imgw):
                for k in range(knc):
                    newDZ[n,
                          i*sh:i*sh+kh,
                          j*sw:j*sw+kw, :] += np.multiply(dZ[n, i, j, k],
                                                          W[..., k])
                    dW[..., k] += np.multiply(dZ[n, i, j, k],
                                              new[n,
                                                  i*sh:i*sh+kh,
                                                  j*sw:j*sw+kw, :])
    if padding == 'same':
        newDZ = newDZ[:, imghp:-imghp, imgwp:-imgwp, :]
    return newDZ, dW, db
