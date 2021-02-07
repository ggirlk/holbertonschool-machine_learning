#!/usr/bin/env python3
""" doc """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ doc """
    m, imgh, imgw, c = A_prev.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    imghp, imgwp = 0, 0
    if padding == 'same':
        imghp = (((imgh - 1) * sh + kh - imgh) // 2) + int(kh % 2 == 0)
        imgwp = (((imgw - 1) * sw + kw - imgw) // 2) + int(kw % 2 == 0)

    if type(padding) == tuple:
        imghp, imgwp = padding
    imgh, imgw = (imgh-kh+2*imghp)//sh + 1, (imgw-kw+2*imgwp)//sw + 1
    output = np.zeros((m, imgh, imgw, knc))
    new = np.pad(A_prev, ((0, 0), (imghp, imghp),
                          (imgwp, imgwp), (0, 0)),
                 'constant', constant_values=0)
    db = np.sum(dZ, axis=(1, 2, 3), keepdims=True)
    newDZ = np.zeros(new.shape)
    dW = np.zeros_like(W)
    for i in range(m):
        for h in range(imgh):
            for w in range(imgw):
                for f in range(c):
                    newDZ[i,
                       h*sh:h*sh+kh,
                       w*sw:(w*sw)+kw,
                       :] += dZ[i, h, w, f] * W[..., f]


                    dW[..., f] += new[i,
                                      h*sh:h*sh+kh,
                                      w*sw:(w*sw)+kw,
                                      :] * dZ[i, h, w, f]

    return newDZ, dW, db
