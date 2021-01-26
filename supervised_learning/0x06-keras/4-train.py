#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K
import tensorflow as tf


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ doc """
    tf.set_random_seed(0)
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)
