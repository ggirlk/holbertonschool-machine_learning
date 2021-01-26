#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    model = K.Sequential()
    freg = K.regularizers.l2(l=lambtha)
    for i in range(len(layers)):
        layer = K.layers.Dense(layers[i],  input_dim=nx,
                                 activation=activations[i],
                                 kernel_regularizer=freg)
        model.add(layer)
        dropped = K.layers.Dropout(rate=keep_prob)
        model.add(dropped)
    return (model)
