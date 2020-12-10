#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if len(poly) == 0 or type(poly) is not list:
        return None
    
    if len(poly) == 1:
        return [0]
    drv = []
    for i in range(1, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        drv.append(poly[i] * i)

    return drv
