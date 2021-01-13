#!/usr/bin/env python3
"""
calculates the normalization
(standardization) constants of a matrix
"""


def normalization_constants(X):
    """ doc """
    return X.mean(axis=0) , X.std(axis=0)
