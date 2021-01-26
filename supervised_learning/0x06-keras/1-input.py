#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """
    inputs = K.Input(shape=(nx,))
    freg = K.regularizers.l2(lambtha)
    x = inputs
    for i in range(0, len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=freg)(x)
        if i != len(layers)-1:
            x = K.layers.Dropout(rate=1-keep_prob)(x)
        
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return (model)