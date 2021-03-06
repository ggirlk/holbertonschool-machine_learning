#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    model = K.Sequential()
    for i in range(len(layers)):
        init = K.initializers.VarianceScaling(mode="fan_avg")
        # freg = K.layers.ActivityRegularization(l2=lambtha)
        freg = K.regularizers.l2(lambtha)
        layer = K.layers.Dense(layers[i], input_dim=nx,
                               activation=activations[i],
                               kernel_initializer=init,
                               kernel_regularizer=freg)
        model.add(layer)
        if i != len(layers)-1:
            dropped = K.layers.Dropout(rate=1-keep_prob)
            model.add(dropped)
    return (model)
