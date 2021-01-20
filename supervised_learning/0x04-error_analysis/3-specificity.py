#!/usr/bin/env python3
""" doc """
import numpy as np


def specificity(confusion):
    """ doc """
    m = confusion.shape[0]
    r = []
    for i in range(m):
        tp = confusion[i][i]
        fn = confusion.T[i].sum()-tp
        tn = confusion.sum() - fn - tp
        allnegatives = confusion.sum() - confusion[i].sum()
        r.append(np.round(tn/allnegatives, 8))
    return np.array(r)
