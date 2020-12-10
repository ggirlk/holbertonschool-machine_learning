#!/usr/bin/env python3
""" doc """


def poly_integral(poly, C=0):
    """ doc """
    if len(poly) == 0 or type(poly) != list:
        return None
    if not isinstance(C, (int, float)):
        return None
    ingrl = [C]
    for i in range(0, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        ingrl.append(poly[i] / (i + 1))

    return ingrl