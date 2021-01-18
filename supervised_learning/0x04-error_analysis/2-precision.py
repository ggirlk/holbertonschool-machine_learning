#!/usr/bin/env python3
""" doc """
import numpy as np


def precision(confusion):
    """ doc """
    true_pos = np.diag(confusion)
    return (np.round(true_pos / np.sum(confusion, axis=0), 8))
