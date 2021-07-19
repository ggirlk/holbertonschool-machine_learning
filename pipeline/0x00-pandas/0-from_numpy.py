#!/usr/bin/env python3
""" Pandas """

import numpy as np
import pandas as pd


def from_numpy(array):
    """
    ****************************************
    creates a pd.DataFrame from a np.ndarray
    ****************************************
    @array: np.ndarray for creation of the pd.DataFrame
    *** The columns of the pd.DataFrame should be labeled
        in alphabetical order and capitalized. There will
        not be more than 26 columns.
    Returns:
            the newly created pd.DataFrame
    """
    a = 65
    alphab = np.array(list(map(chr,
                               range(a, a + array.shape[1]))))
    df = pd.DataFrame(array)
    df.columns = alphab
    return df
