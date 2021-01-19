#!/usr/bin/env python3
""" doc """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ doc """
    m = labels.shape[0] # m 
    k = labels.shape[1] # Number of classes 
    result = np.zeros((k, k))
    y_pred = np.argmax(logits, axis=1)
    y_true = np.argmax(labels, axis=1)
    ind = np.logical_and(y_pred < k, y_true < k)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = np.ones(m, dtype=np.int64)
    sample_weight = sample_weight[ind]
    for i in range(m):
        a, b = y_true[i], y_pred[i]
        result[a][b] += sample_weight[i]
    return (result)
