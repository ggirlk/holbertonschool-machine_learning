#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if len(poly) == 0 or type(poly) != list:
        return None

    drv = [0]
    for i in range(1, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        drv.append(poly[i] * i)
    if sum(drv) == 0:
        return [0]
    del drv[0]
    return drv
