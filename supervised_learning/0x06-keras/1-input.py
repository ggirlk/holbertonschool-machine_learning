#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    inputs = K.Input(shape=(nx,))
    freg = K.regularizers.l2(lambtha)
    x = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=freg)(inputs)
    for i in range(1, len(layers)):
        x = K.layers.Dropout(rate=1-keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=freg)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return (model)
