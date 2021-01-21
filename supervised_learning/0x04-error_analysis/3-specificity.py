#!/usr/bin/env python3
""" doc """
import numpy as np


def specificity(confusion):
    """ doc """
    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_neg = np.sum(confusion, axis=1) - true_pos
    true_neg = np.sum(confusion) - false_neg - true_pos - false_pos
    return (true_neg / (true_neg + false_pos))
