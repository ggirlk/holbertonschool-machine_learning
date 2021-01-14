#!/usr/bin/env python3
"""
calculates the weighted
moving average of a data set
"""


def def moving_average(data, beta):
    """ doc """
    m_avg = []
    vt = 0
    for i in range(len(data)):
        vt = (vt*beta + data[i]*(1-beta))
        avg = vt/(1-beta**(i+1))
        m_avg.append(avg)
    return m_avg
