#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ doc """
    return K.utils.to_categorical(labels, classes)
