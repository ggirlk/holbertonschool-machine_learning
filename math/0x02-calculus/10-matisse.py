#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if len(poly) == 0 or type(poly) != list:
        None
    drv = []
    for i in range(1, len(poly)):
        if type(poly[i]) != int and type(poly[i]) != float:
            return None
        drv.append(poly[i] * i)

    if len(poly) == 1 or sum(drv) == 0:
        return [0]
    return drv
