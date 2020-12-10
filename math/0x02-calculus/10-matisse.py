#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if type(poly) != list:
        None
    drv = []
    for i in range(1, len(poly)):
        drv.append(poly[i] * i)
    return drv