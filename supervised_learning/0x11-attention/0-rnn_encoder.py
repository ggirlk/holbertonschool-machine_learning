#!/usr/bin/env python3
""" RNN Encoder """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ encoder for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """
        *********************************************
        *****************Constructor*****************
        *********************************************
        @vocab: is an integer representing the size
                of the input vocabulary
        @embedding: is an integer representing the
                    dimensionality of the embedding vector
        @units: is an integer representing the number
                of hidden units in the RNN cell
        @batch: is an integer representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.vocab = vocab
        # keras Embedding layer that converts words from
        # the vocabulary into an embedding vector
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.batch = batch
        # a keras GRU layer with units units
        self.gru = tf.keras.layers.GRU(
                    units,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="glorot_uniform",
                    return_sequences=True,
                    return_state=True
                    )

    def initialize_hidden_state(self):
        """
        nitializes the hidden states for
        the RNN cell to a tensor of zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        calls the encoders layers
        @x: is a tensor of shape (batch, input_seq_len) containing
            the input to the encoder layer as word indices
            within the vocabulary
        @initial: is a tensor of shape (batch, units) containing
                  the initial hidden state
        Returns:
                outputs: is a tensor of shape (batch, input_seq_len,
                         units) containing the outputs of the encoder
                hidden: is a tensor of shape (batch, units) containing
                        the last hidden state of the encoder
        """
        embading = self.embedding(x)
        outputs = self.gru(embading, initial_state=initial)
        return outputs
