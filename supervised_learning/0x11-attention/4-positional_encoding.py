#!/usr/bin/env python3
""" Positional Encoding """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    calculates the positional encoding for a transformer
    @max_seq_len: is an integer representing the maximum
                  sequence length
    @dm: is the model depth
    Returns:
            a numpy.ndarray of shape (max_seq_len, dm)
            containing the positional encoding vectors
    """
    pEncoding = np.ndarray((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(dm):
            if j % 2:
                pEncoding[i][j] = np.cos(i / np.power(10000, (j - 1) / dm))
            else:
                pEncoding[i][j] = np.sin(i / np.power(10000, j / dm))
    return pEncoding
