#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ doc """
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
    return None
