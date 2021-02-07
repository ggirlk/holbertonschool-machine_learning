#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K


def lenet5(X):
    """ doc """
    layer = K.layers.Conv2D(6, 5, padding='same',
                            activation='relu',
                            kernel_initializer='he_normal')(X)
    layer = K.layers.MaxPooling2D(2, 2)(layer)
    layer = K.layers.Conv2D(16, 5, padding='valid', activation='relu',
                            kernel_initializer='he_normal')(layer)
    layer = K.layers.MaxPooling2D(2, 2)(layer)
    layer = K.layers.Flatten()(layer)
    layer = K.layers.Dense(120, activation='relu',
                           kernel_initializer='he_normal')(layer)
    layer = K.layers.Dense(84, activation='relu',
                           kernel_initializer='he_normal')(layer)
    layer = K.layers.Dense(10, activation='softmax',
                           kernel_initializer='he_normal')(layer)
    model = K.Model(X, layer)
    model.compile('Adam', metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model
