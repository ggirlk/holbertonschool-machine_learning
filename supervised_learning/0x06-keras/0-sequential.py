#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    model = K.Sequential()
    dropped = K.layers.Dropout(rate=keep_prob)
    freg = K.regularizers.l2(l=lambtha)
    for i in range(len(layers)):
        model.add(K.layers.Dense(layers[i], input_dim=nx,
                                 activation=activations[i],
                                 kernel_regularizer=freg,
                                 activity_regularizer=dropped))
    return (model)