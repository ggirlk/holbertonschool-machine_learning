#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ doc """
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Conv2D = K.layers.Conv2D
    Dropout = K.layers.Dropout(rate=0.4)
    Dense = K.layers.Dense
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation
    Add = K.layers.Add

    def layersConv(X, k, f, s=None, p='same'):
        layer = Conv2D(k, f, s, padding=p,
                       kernel_initializer='he_normal')(X)
        layer = BatchNorm()(layer)
        layer = Activation('relu')(layer)
        return layer

    X = K.Input(shape=(224, 224, 3))

    layer = layersConv(X, 64, 7, 2)

    layerMax = MaxPooling2D(3, 1, padding='same')(layer)

    layer = projection_block(layer, [64, 64, 256])
    layer = identity_block(layer, [64, 64, 256])
    layer = identity_block(layer, [64, 64, 256])

    layer = projection_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])

    layer = projection_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])

    layer = projection_block(layer, [512, 512, 2048])
    layer = identity_block(layer, [512, 512, 2048])
    layer = identity_block(layer, [512, 512, 2048])

    layerAVG = AveragePooling2D(1, 1)(layer)

    Y = Dense(1000, activation="softmax",
              kernel_initializer='he_normal')(layerAVG)

    model = K.Model(inputs=X, outputs=Y)
    return model
