#!/usr/bin/env python3
""" summation n squared """


def summation_i_squared(n):
    """ summation n squared """
    ls = list(range(1, n+1))
    s = sum(map(lambda x: pow(x, 2), ls))
    return s
