#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """ doc """
    cache = {}
    cache["A0"] = X
    for i in range(1, L):
        W = tf.cast(weights["W"+str(i)], dtype="float32")
        b = tf.cast(weights["b"+str(i)], dtype="float32")
        dropped = tf.nn.dropout(cache['A'+str(i-1)], keep_prob)
        print(W.shape, b.shape, dropped.shape)
        dense = np.dot(dropped, tf.transpose(W)) + b
        cache['A'+str(i)] = tf.nn.softmax(dense)
    i += 1
    W = tf.cast(weights["W"+str(i)], dtype="float32")
    b = tf.cast(weights["b"+str(i)], dtype="float32")
    dropped = tf.layers.Dropout(cache['A'+str(i-1)], keep_prob)
    dense = np.dot(dropped, tf.transpose(W))+b
    cache['A'+str(i)] = tf.nn.tanh(dense)
    return cache
