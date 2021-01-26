#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    model = K.Sequential()
    freg = K.layers.ActivityRegularization(l2=lambtha)
    layer = K.layers.Dense(layers[0], input_dim=nx,
                           activation=activations[0],
                           kernel_regularizer=freg)
    model.add(layer)
    dropped = K.layers.Dropout(rate=keep_prob)
    model.add(dropped)
    for i in range(1, len(layers)):
        freg = K.layers.ActivityRegularization(l2=lambtha)
        layer = K.layers.Dense(layers[i],
                               activation=activations[i],
                               kernel_regularizer=freg)
        model.add(layer)
        #if i != len(layers)-1:
        dropped = K.layers.Dropout(rate=keep_prob)
        model.add(dropped)
    return (model)
