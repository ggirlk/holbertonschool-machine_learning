#!/usr/bin/env python3
""" summation n squared """


def summation_i_squared(n):
    """ summation n squared """
    ls = list(range(n, n))
    return sum(map(lambda x: pow(x, 2), ls))
