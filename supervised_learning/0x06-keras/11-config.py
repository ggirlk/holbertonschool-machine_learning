#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def save_config(network, filename):
    """ doc """
    network.to_json(filename)


def load_config(filename):
    """ doc """
    K.models.model_from_json(filename)
