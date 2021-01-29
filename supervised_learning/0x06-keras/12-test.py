#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ doc """
    return network.predict(data,
                           verbose=verbose)
