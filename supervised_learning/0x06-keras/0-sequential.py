#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    model = K.Sequential()
    init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
    freg = K.regularizers.l2(l=lambtha)
    layer = K.layers.Dense(layers[0], input_shape=(nx,),
                           activation=activations[0],
                           kernel_initializer=init,
                           kernel_regularizer=freg)
    model.add(layer)
    dropped = K.layers.Dropout(rate=keep_prob)
    model.add(dropped)
    for i in range(1, len(layers)):
        init = (tf.contrib.layers.
            variance_scaling_initializer(mode="FAN_AVG"))
        freg = K.regularizers.l2(l=lambtha)
        layer = K.layers.Dense(layers[i], input_shape=(nx,),
                               activation=activations[i],
                               kernel_initializer=init,
                               kernel_regularizer=freg)
        model.add(layer)
        if i != len(layers)-1:
            dropped = K.layers.Dropout(rate=keep_prob)
            model.add(dropped)
    return (model)
