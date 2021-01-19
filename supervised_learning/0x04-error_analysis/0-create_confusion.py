#!/usr/bin/env python3
""" doc """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ doc """
    m = labels.shape[0]
    k = labels.shape[1]
    result = np.zeros((k, k))
    y_pred = np.argmax(logits, axis=1)
    y_true = np.argmax(labels, axis=1)
    for i in range(m):
        a, b = y_true[i], y_pred[i]
        result[a][b] += 1
    return (result)
