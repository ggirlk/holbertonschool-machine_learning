#!/usr/bin/env python3
""" Multi Head Attention """
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ perform multi head attention """

    def __init__(self, dm, h):
        """
        *********************************************
        *****************Constructor*****************
        *********************************************
        @dm: is an integer representing the dimensionality
             of the model (divisible by h)
        @h: is an integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        # the number of heads
        self.h = h
        # the dimensionality of the model
        self.dm = dm
        # the depth of each attention head
        self.depth = dm // h
        # a Dense layer with dm units, used to generate the query matrix
        self.Wq = tf.keras.layers.Dense(dm)
        # a Dense layer with dm units, used to generate the key matrix
        self.Wk = tf.keras.layers.Dense(dm)
        # a Dense layer with dm units, used to generate the value matrix
        self.Wv = tf.keras.layers.Dense(dm)
        # a Dense layer with dm units, used to generate the attention output
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        ************************************************
        ****************Keras layer call****************
        ************************************************
        @Q: is a tensor of shape (batch, seq_len_q, dk)
            containing the input to generate the query matrix
        @K: is a tensor of shape (batch, seq_len_v, dk)
            containing the input to generate the key matrix
        @V: is a tensor of shape (batch, seq_len_v, dv)
            containing the input to generate the value matrix
        @mask: is always None
        Returns:
                output: a tensor with its last two dimensions as
                        (..., seq_len_q, dm) containing the scaled
                        dot product attention
                weights: a tensor with its last three dimensions as
                        (..., h, seq_len_q, seq_len_v) containing
                        the attention weights
        """
        batches = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        def split_heads(x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        Q = split_heads(Q, batches)
        K = split_heads(K, batches)
        V = split_heads(V, batches)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batches, -1, self.dm])
        return self.linear(output), weights
