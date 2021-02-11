#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ doc """
    Conv2D = K.layers.Conv2D
    BatchNorm = K.layers.BatchNormalization
    act = K.layers.Activation
    add = K.layers.Add
    F11, F3, F12 = filters
    layer1x1 = Conv2D(F11, 1, 1, padding='same')(A_prev)
    layer1x1 = BatchNorm()(layer1x1)
    layer1x1 = act('relu')(layer1x1)

    layer3x3 = Conv2D(F3, 1, 1, padding='same')(layer1x1)
    layer3x3 = BatchNorm()(layer3x3)
    layer3x3 = act('relu')(layer3x3)

    layer1x1 = Conv2D(F12, 4, 1, padding='same')(layer3x3)
    layer1x1 = BatchNorm()(layer1x1)

    layer_out = add()([layer1x1, A_prev])
    layer_out = act('relu')(layer_out)

    return layer_out
