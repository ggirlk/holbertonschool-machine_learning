#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ doc """
    Conv2D = K.layers.Conv2D
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Add = K.layers.Add
    F11, F3, F12 = filters

    layer1x1 = Conv2D(F11, 1, s,padding='same',
                      kernel_initializer='he_normal')(A_prev)
    layer1x1 = BatchNorm()(layer1x1)
    layer1x1 = Activation('relu')(layer1x1)

    layer3x3 = Conv2D(F3, 3, padding='same',
                      kernel_initializer='he_normal')(layer1x1)
    layer3x3 = BatchNorm()(layer3x3)
    layer3x3 = Activation('relu')(layer3x3)

    layer1x1 = Conv2D(F12, 1,padding='same',
                      kernel_initializer='he_normal')(layer3x3)
    layer1x1 = BatchNorm()(layer1x1)  #

    layer1x1_s = Conv2D(F12, 1, s,padding='same',
                        kernel_initializer='he_normal')(A_prev)
    layer1x1_s = BatchNorm()(layer1x1_s)  #

    layer_out = Add()([layer1x1, layer1x1_s])
    layer_out = Activation('relu')(layer_out)

    return layer_out
