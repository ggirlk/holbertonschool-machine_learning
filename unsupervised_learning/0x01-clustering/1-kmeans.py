#!/usr/bin/env python3
""" performs K-means on a dataset """

import numpy as np


def initialize(X, k):
    """ doc """
    try:
        n, d = X.shape
        if k == 0 or d == 0:
            return None
        return np.random.uniform(np.amin(X, axis=0),
                                 np.amax(X, axis=0),
                                 size=(k, d))
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """ doc """
	try:
		n, d = X.shape
		if type(iterations) is not int or iterations < 1:
			return None, None
		C = initialize(X, k)
		if C is None:
			return None, None
		clss = None
		for itr in range(iterations):
			prevC = C.copy()
			clss = np.apply_along_axis(np.subtract, 1, X, C)
			clss = np.argmin(np.square(clss).sum(axis=2), axis=1)
			for cent in range(k):
				Xs = np.argwhere(clss == cent)
				if Xs.shape[0] == 0:
					C[cent] = initialize(X, 1)
				else:
					C[cent] = np.mean(X[Xs], axis=0)
			if np.all(prevC == C):
				break;
		return C, clss
	except Exception:
        return None, None
