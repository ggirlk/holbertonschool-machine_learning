#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if len(poly) == 0 or type(poly) is not list:
        return None

    drv = []
    for i in range(0, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        drv.append(poly[i] * i)
    if len(poly) == 0 or sum(drv) == 0:
        return [0]
    del drv[0]
    return drv
