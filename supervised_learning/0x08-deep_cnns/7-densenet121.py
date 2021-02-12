#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ doc """
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dense = K.layers.Dense
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation

    def layersConv(X, k, f, s=None, p='same'):
        layer = Conv2D(k, f, s, padding=p,
                       kernel_initializer='he_normal')(X)
        layer = BatchNorm()(layer)
        layer = Activation('relu')(layer)
        return layer

    X = K.Input(shape=(224, 224, 3))
    nb_filters = 64
    # Convolution
    layer = layersConv(X, nb_filters, 7, 2)

    # Pooling
    layerMax = MaxPooling2D(3, 2, padding="same")(layer)

    # Dense Block / Transition Layer (1)
    layers = 6
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, layers)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)

    # Dense Block / Transition Layer (2)
    layers = 12
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, layers)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)

    # Dense Block / Transition Layer (3)
    layers = 24
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, layers)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)

    # Dense Block (3)
    layers = 16
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, layers)

    # Classification Layer (Pooling + 1000D fully-connected, softmax)
    layerAVG = AveragePooling2D(7, 7, padding="same")(layer)
    Y = Dense(1000, activation="softmax")(layerAVG)

    model = K.Model(inputs=X, outputs=Y)
    return model
