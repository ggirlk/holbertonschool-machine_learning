#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if type(poly) != list:
        None
    drv = []
    for i in range(1, len(poly)):
        if type(poly[i]) != int:
            return None
        drv.append(poly[i] * i)
    if sum(drv) == 0 or len(poly) == 1:
        retun [0]
    return drv
