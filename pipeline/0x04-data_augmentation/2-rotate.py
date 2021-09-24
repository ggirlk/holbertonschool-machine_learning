#!/usr/bin/env python3
"""Rotate"""
import tensorflow as tf


def rotate_image(image):
    """rotates an image by 90 degrees counter-clockwise"""
    return tf.image.rot90(image)
