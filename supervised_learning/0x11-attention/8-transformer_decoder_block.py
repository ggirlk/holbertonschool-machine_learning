#!/usr/bin/env python3
"""Transformer Decoder Block"""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer Decoder Block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
         """
        *********************************************
        *****************Constructor*****************
        *********************************************
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in
                 the fully connected layer
        @drop_rate: the dropout rate
        """
        super(DecoderBlock, self).__init__()
        #  the first MultiHeadAttention layer
        self.mha1 = MultiHeadAttention(dm, h)
        # the second MultiHeadAttention layer
        self.mha2 = MultiHeadAttention(dm, h)
        # the hidden dense layer with hidden units and relu activation
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # the output dense layer with dm units
        self.dense_output = tf.keras.layers.Dense(dm)
        # the first layer norm layer, with epsilon=1e-6
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the second layer norm layer, with epsilon=1e-6
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the third layer norm layer, with epsilon=1e-6
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # the first dropout layer
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # the second dropout layer
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        # the third dropout layer
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        ************************************************
        ****************Keras layer call****************
        ************************************************
        @x: a tensor of shape (batch, target_seq_len, dm) containing
            the input to the decoder block
        @encoder_output: a tensor of shape (batch, input_seq_len, dm)
                         containing the output of the encoder
        @training: a boolean to determine if the model is training
        @look_ahead_mask: the mask to be applied to the first multi
                          head attention layer
        @padding_mask: the mask to be applied to the second multi head
                       attention layer
        Returns:
                a tensor of shape (batch, target_seq_len, dm)
                containing the block’s output
        """
        start, weights1 = self.mha1(x, x, x, look_ahead_mask)
        start = self.dropout1(start, training=training)
        start = self.layernorm1(x + start)
        mid, weights2 = self.mha2(start, encoder_output, encoder_output,
                                  padding_mask)
        mid = self.dropout2(mid, training=training)
        mid = self.layernorm2(start + mid)
        out = self.dense_hidden(mid)
        out = self.dense_output(out)
        out = self.dropout3(out, training=training)
        out = self.layernorm3(mid + out)
        return out
