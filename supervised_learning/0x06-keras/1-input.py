#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    inputs = K.Input(shape=(nx,))
    init = K.initializers.VarianceScaling(mode="fan_avg")
    freg = K.layers.ActivityRegularization(l2=lambtha)
    dropped = K.layers.Dropout(rate=keep_prob)
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_initializer=init,
                       kernel_regularizer=freg)(inputs)
    for i in range(1, len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_initializer=init,
                           kernel_regularizer=freg)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.Dropout = dropped
    return (model)