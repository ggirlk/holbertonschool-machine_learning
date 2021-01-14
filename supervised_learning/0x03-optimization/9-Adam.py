#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ doc """
    
