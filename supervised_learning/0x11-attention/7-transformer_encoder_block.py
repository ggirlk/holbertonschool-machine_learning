#!/usr/bin/env python3
"""Transformer Encoder Block"""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer Encoder Block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        *********************************************
        *****************Constructor*****************
        *********************************************
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the fully connected layer
        @drop_rate: the dropout rate
        """
        super(EncoderBlock, self).__init__()
        # MultiHeadAttention layer
        self.mha = MultiHeadAttention(dm, h)
        # the hidden dense layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # the output dense layer with dm units
        self.dense_output = tf.keras.layers.Dense(dm)
        # the first layer norm layer, with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the second layer norm layer, with epsilon=1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #  the first dropout layer
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # the second dropout layer
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        ************************************************
        ****************Keras layer call****************
        ************************************************
        @x: a tensor of shape (batch, input_seq_len, dm) containing
            the input to the encoder block
        @training: a boolean to determine if the model is training
        @mask: the mask to be applied for multi head attention
        Returns:
                a tensor of shape (batch, input_seq_len, dm) containing
                the block’s output
        """
        mha, w = self.mha(x, x, x, mask)
        mha = self.dropout1(mha, training=training)
        mha = self.layernorm1(x + mha)
        output = self.dense_hidden(mha)
        output = self.dense_output(output)
        output = self.dropout2(output, training=training)
        output = self.layernorm2(mha + output)
        return output
