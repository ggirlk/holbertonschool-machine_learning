#!/usr/bin/env python3
""" element-wise addition, subtraction, multiplication, division """


def np_elementwise(mat1, mat2):
    """ element-wise addition, subtraction, multiplication, division """
    import numpy as np
    return [np.add(mat1, mat2), np.subtract(mat1, mat2),
            np.multiply(mat1, mat2), np.divide(mat1, mat2)]
