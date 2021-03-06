#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
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

    layer_new = BatchNorm()(X)
    layer_new = Activation('relu')(layer_new)
    nb_filters = int(nb_filters * compression)
    layer_new = layersConv(layer_new, nb_filters, 1)
    layer_new = AveragePooling2D(2)(layer_new)

    return layer_new, nb_filters
