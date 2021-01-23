#!/usr/bin/env python3
""" doc """
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ doc """
    recall = sensitivity(confusion)
    pr = precision(confusion)
    return (recall*pr)/(recall+pr)
