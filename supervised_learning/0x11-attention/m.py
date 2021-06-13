#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

np.random.seed(0)
tf.set_random_seed(0)
decoder = RNNDecoder(2048, 128, 256, 32)
print(type(decoder.embedding), decoder.embedding.input_dim, decoder.embedding.output_dim)
print(type(decoder.gru), decoder.gru.units)
print(type(decoder.F), decoder.F.units)

with open('1-test', 'w+') as f:
    x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
    s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
    hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
    y, s = decoder(x, s_prev, hidden_states)
    Y = tf.keras.backend.eval(y)
    S = tf.keras.backend.eval(s)
    f.write(str(Y.shape) + '\n' + np.array2string(Y, precision=5) + '\n')
    f.write(str(S.shape) + '\n' + np.array2string(S, precision=5) + '\n')
