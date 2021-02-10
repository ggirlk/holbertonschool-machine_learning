#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def identity_block(A_prev, filters):
    """ doc """
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dropout = K.layers.Dropout(rate=0.4)
    Dense = K.layers.Dense
    Concatenate = K.layers.Concatenate
    BatchNorm = K.layers.BatchNormalization
    relu = K.layers.ReLU
    add = K.layers.Add
    F11, F3, F12 = filters
    layer1x1 = Conv2D(F11, 1, 1, padding='same',  activation='relu')(A_prev)
    layer1x1 = BatchNorm()(layer1x1)
    layer3x3 = Conv2D(F3, 1, 1, padding='same',  activation='relu')(layer1x1)
    layer3x3 = BatchNorm()(layer3x3)
    layer1x1 = Conv2D(F12, 4, 1, padding='same')(layer3x3)
    layer1x1 = BatchNorm()(layer1x1)
    layer_out = add()([layer1x1, A_prev])
    #layer_out = relu(layer_out)
    return layer_out
