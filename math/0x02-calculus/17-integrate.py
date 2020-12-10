#!/usr/bin/env python3
""" doc """


def poly_integral(poly, C=0):
    """ doc """
    if len(poly) == 0 or type(poly) != list:
        return None
    if not isinstance(C, int):
        return None
    ingrl = [C]
    if (len(poly) == 1):
        return ingrl
    for i in range(0, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        it = poly[i] / (i + 1)
        it = it.__trunc__() if not it % 1 else float(it)
        ingrl.append(it)

    return ingrl
