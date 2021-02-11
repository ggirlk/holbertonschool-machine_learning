#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ doc """
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dense = K.layers.Dense
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Concatenate = K.layers.Concatenate

    def layersConv(X, k, f, p='valid', s=1):
        layer = Conv2D(k, f, s, padding=p,
                       kernel_initializer='he_normal')(X)
        return layer

    layer_prev = X
    for layer in range(layers):
        layer_new = BatchNorm()(layer_prev)
        layer_new = Activation('relu')(layer_new)
        layer_new = layersConv(layer_new, growth_rate, nb_filters, 'same')
        layer_new = BatchNorm()(layer_new)
        layer_new = Activation('relu')(layer_new)
        layer_new = layersConv(layer_new, growth_rate, nb_filters, 'same')
        layer_prev = Concatenate()([layer_prev, layer_new])
        nb_filters += growth_rate
    return layer_prev, nb_filters
