#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def l2_reg_cost(cost):
    """ doc """
    
    return tf.contrib.losses.compute_weighted_loss(cost)
