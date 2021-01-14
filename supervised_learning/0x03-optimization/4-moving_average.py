#!/usr/bin/env python3
"""
calculates the weighted
moving average of a data set
"""


def def moving_average(data, beta):
    """ doc """
    m_avg = []
    for i in range(len(data)):
        if i == 0:
            avg = data[i]
        else:
            d = 0
            coef = 1
            for j in range(i):
                d += data[j]*beta
                coef += beta
            # print(data[j], data[i])
            # print("-----", coef)
            avg = (d + data[i])/(coef)
        m_avg.append(avg)
    return m_avg
