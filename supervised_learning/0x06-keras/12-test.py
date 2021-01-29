#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ doc """
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
