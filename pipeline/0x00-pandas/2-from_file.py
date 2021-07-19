#!/usr/bin/env python3
""" From File """
import numpy as np
import pandas as pd


def from_file(filename, delimiter):
    """
    ****************************************
    loads data from a file as a pd.DataFrame
    ****************************************
    @filename: is the file to load from
    @delimiter: is the column separator
    Returns:
            the loaded pd.DataFrame
    """
    with open(filename, 'r') as f:
        headers = list(f.readline().strip().split(delimiter))
        body = []
        for line in f.readlines():
            body.append(np.array(line.strip().split(delimiter)))
    df = pd.DataFrame(np.array(body))
    df.columns = headers
    return df
