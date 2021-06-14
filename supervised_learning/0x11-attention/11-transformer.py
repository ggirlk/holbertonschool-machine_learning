#!/usr/bin/env python3
"""Transformer Network"""


import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """Transformer Network"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        *********************************************
        *****************Constructor*****************
        *********************************************
        @N: the number of blocks in the encoder and decoder
        @dm: the dimensionality of the model
        @h: the number of heads
        @hidden: the number of hidden units in the fully connected layers
        @input_vocab: the size of the input vocabulary
        @target_vocab: the size of the target vocabulary
        @max_seq_input: the maximum sequence length possible for the input
        @max_seq_target: the maximum sequence length possible for the target
        @drop_rate: the dropout rate
        """
        super(Transformer, self).__init__()
        # the encoder layer
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate=0.1)
        # the decoder layer
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate=0.1)
        # final Dense layer with target_vocab units
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        ************************************************
        ****************Keras layer call****************
        ************************************************
        @inputs: a tensor of shape (batch, input_seq_len)containing the inputs
        @target: a tensor of shape (batch, target_seq_len)containing the target
        @training: a boolean to determine if the model is training
        @encoder_mask: the padding mask to be applied to the encoder
        @look_ahead_mask: the look ahead mask to be applied to the decoder
        @decoder_mask: the padding mask to be applied to the decoder
        Returns:
                a tensor of shape (batch, target_seq_len, target_vocab)
                containing the transformer output
        """
        enc_out = self.encoder(inputs, training, encoder_mask)
        dec_out = self.decoder(target, enc_out, training, look_ahead_mask,
                               decoder_mask)
        return self.linear(dec_out)
