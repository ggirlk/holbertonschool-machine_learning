#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(0, len(layers)):
        init = K.initializers.VarianceScaling(mode="fan_avg")
        freg = K.regularizers.l2(lambtha)
        dropped = K.layers.Dropout(rate=1-keep_prob)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_initializer=init,
                           kernel_regularizer=freg)(x)
        if i != len(layers)-1:
            x = dropped(x)
    model = K.Model(inputs=inputs, outputs=x)
    return (model)
