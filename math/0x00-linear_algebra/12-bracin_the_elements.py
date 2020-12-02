#!/usr/bin/env python3
""" element-wise addition, subtraction, multiplication, division """


def np_elementwise(mat1, mat2):
    """ element-wise addition, subtraction, multiplication, division """
    return [mat1 + mat2, mat1 - mat2,
            mat1 * mat2, mat1 / mat2]
