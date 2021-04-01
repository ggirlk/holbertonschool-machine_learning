#!/usr/bin/env python3
""" initialize cluster centroids for K-means """

import numpy as np


def initialize(X, k):
    """ doc """
    try:
		n, d = X.shape
        return np.random.uniform(X.min(axis=0),
								 X.max(axis=0),
								 size=(k, d))
    except Exception:
        return None
