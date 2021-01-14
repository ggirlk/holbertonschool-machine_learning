#!/usr/bin/env python3
"""
training operation for a neural network
in tensorflow using the gradient descent
with momentum optimization algorithm
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ doc """

    return alpha/(1+(decay_rate*(global_step)))
