#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ doc """
    if filename[-2:] != save_format:
        filename += save_format
    network.save_weights(filename)


def load_weights(network, filename):
    """ doc """
    return network.load_weights(filename)
