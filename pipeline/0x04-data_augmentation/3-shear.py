#!/usr/bin/env python3
"""Shear"""
import tensorflow as tf


def shear_image(image, intensity):
    """randomly shears an image"""
    return tf.keras.preprocessing.image.random_shear(image.numpy(), intensity,
                                                     channel_axis=2)
