#!/usr/bin/env python3
""" RNN Decoder """
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ decode for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """
        *********************************************
        *****************Constructor*****************
        *********************************************
        @vocab: is an integer representing the size of
                the output vocabulary
        @embedding: is an integer representing the
                    dimensionality of the embedding vector
        @units: is an integer representing the number
                of hidden units in the RNN cell
        @batch: is an integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        # keras Embedding layer that converts words from
        # the vocabulary into an embedding vector
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # keras GRU layer with units units
        # return both the full sequence of output
        # as well as the last hidden state
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True)
        # Dense layer with vocab units
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        **********************************************************
        *****************calls the decoder layers*****************
        **********************************************************
        @x: is a tensor of shape (batch, 1) containing the previous
            word in the target sequence as an index of the target vocabulary
        @s_prev: is a tensor of shape (batch, units) containing the
                 previous decoder hidden state
        @hidden_states: is a tensor of shape (batch, input_seq_len,
                        units)containing the outputs of the encoder
        Returns:
                y: is a tensor of shape (batch, vocab) containing
                   the output word as a one hot vector in the target vocabulary
                s: is a tensor of shape (batch, units) containing
                   the new decoder hidden state
        """
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], -1)
        output, s = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, s
