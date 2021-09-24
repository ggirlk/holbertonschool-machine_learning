"""
PCA Color Augmentation
https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """ performs PCA color augmentation """
    img = np.reshape(image, (image.shape[0] * image.shape[1], 3))

    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)

    img = img.astype('float32')
    img -= np.mean(img, axis=0)
    img /= np.std(img, axis=0)

    cov = np.cov(img, rowvar=False)

    lambdas, p = np.linalg.eig(cov)

    delta = np.dot(p, alphas * lambdas)

    pca_augmentation = img + delta
    pca_color_image = pca_augmentation * std + mean
    pca_color_image = pca_color_image.reshape(image.shape[0], image.shape[1], 3)
    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype(
        'uint8')

    return pca_color_image
