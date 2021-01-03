#!/usr/bin/env python3
"""
converts a numeric label
vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ doc """

    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
