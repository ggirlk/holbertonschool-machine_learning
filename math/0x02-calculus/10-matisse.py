#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if type(poly) != list:
        None
    if len(poly) == 1:
        return [0]
    drv = []
    for i in range(1, len(poly)):
        if type(poly[i]) != int or type(poly[i]) != float:
            return None
        drv.append(poly[i] * i)
    return drv
