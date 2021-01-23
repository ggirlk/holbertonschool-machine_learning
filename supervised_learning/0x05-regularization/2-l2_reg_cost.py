#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def l2_reg_cost(cost):
    """ doc """

    return cost+tf.losses.get_regularization_losses()
