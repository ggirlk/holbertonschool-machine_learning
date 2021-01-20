#!/usr/bin/env python3
""" doc """
import numpy as np


def specificity(confusion):
    """ doc """
    m = confusion.shape[0]
    r = []
    for i in range(m):
        tp = confusion[i][i]
        fp = confusion.T[i].sum()-tp
        tn = confusion.sum() - fp - tp
        r.append(np.round(tn/(tn+fp), 8))
    return np.array(r)
