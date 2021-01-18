#!/usr/bin/env python3
""" doc """
import numpy as np


def sensitivity(confusion):
    """ doc """
    r = []
    for i in confusion:
        r.append(np.round(i.max()/i.sum(), 8))
    return(np.array(r))
