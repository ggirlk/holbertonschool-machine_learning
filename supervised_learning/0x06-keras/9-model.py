#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def save_model(network, filename):
    """ doc """
    network.save(filename)


def load_model(filename):
    """ doc """
    return K.models.load_model(filename)
