#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ doc """
    if verbose == True:
        verbose = 1
    else:
        verbose = 0
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)
