#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ doc """
    Conv2D = K.layers.Conv2D
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Add = K.layers.Add
    F11, F3, F12 = filters
    layer1x1 = Conv2D(F11, 1, padding='same')(A_prev)
    layer1x1 = BatchNorm()(layer1x1)
    layer1x1 = Activation('relu')(layer1x1)

    layer3x3 = Conv2D(F3, 3, padding='same')(layer1x1)
    layer3x3 = BatchNorm()(layer3x3)
    layer3x3 = Activation('relu')(layer3x3)

    layer1x1 = Conv2D(F12, 1, padding='same')(layer3x3)
    layer1x1 = BatchNorm()(layer1x1)

    layer_out = Add()([layer1x1, A_prev])
    layer_out = Activation('relu')(layer_out)

    return layer_out
