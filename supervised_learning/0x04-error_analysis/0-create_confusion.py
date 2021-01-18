#!/usr/bin/env python3
""" doc """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ doc """
    m = labels.shape[0] # m 
    k = labels.shape[1] # Number of classes 
    result = np.zeros((k, k))
    pred =  np.argmax(logits, axis=1)
    true = np.argmax(labels, axis=1)
    for i in range(m):
        for j in range(k):
            a, b = true[i], pred[i]
            result[a][b] += 1
    return (result)