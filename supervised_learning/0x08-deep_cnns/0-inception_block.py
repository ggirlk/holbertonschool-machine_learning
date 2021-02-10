#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ doc """
    MaxPooling2D = K.layers.MaxPooling2D
    Conv2D = K.layers.Conv2D
    Concatenate = K.layers.Concatenate()

    F1, F3R, F3, F5R, F5, FPP = filters

    layer1x1_0 = Conv2D(F1, 1, activation='relu')(A_prev)  #

    layer1x1_1 = Conv2D(F3R, 1, padding='same', activation='relu')(A_prev)

    layer3x3 = Conv2D(F3, 3, padding='same', activation='relu')(layer1x1_1)  #

    layer1x1_2 = Conv2D(F5R, 1, padding='same', activation='relu')(A_prev)

    layer5x5 = Conv2D(F5, 5, padding='same',  activation='relu')(layer1x1_2)  #

    layerMax = MaxPooling2D(1)(A_prev)

    layer1x1_3 = Conv2D(FPP, 1, padding='same', activation='relu')(layerMax)  #

    layer_out = Concatenate([layer1x1_0, layer3x3, layer5x5, layer1x1_3])
    return layer_out
