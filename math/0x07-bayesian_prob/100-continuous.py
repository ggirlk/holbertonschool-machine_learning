#!/usr/bin/env python3
""" Bayesian Probability """
import scipy.special as special


def posterior(x, n, p1, p2):
    """
    ***************************************************************
    ***calculates the posterior probability that the probability***
    ***of developing severe side effects falls within a specific***
    *********************range given the data**********************
    ***************************************************************
    @x: is the number of patients that develop severe side effects
    @n: is the total number of patients observed
    @p1: is the lower bound on the range
    @p2: is the upper bound on the range
    *** You can assume the prior beliefs of p follow a uniform distribution
    *** If n is not a positive integer, raise a ValueError with the message
        n must be a positive integer
    *** If x is not an integer that is greater than or equal to 0,
        raise a ValueError with the message x must be an integer
        that is greater than or equal to 0
    *** If x is greater than n, raise a ValueError with the message
        x cannot be greater than n
    *** If p1 or p2 are not floats within the range [0, 1], raise a
        ValueError with the message {p} must be a float in the range [0, 1]
        where {p} is the corresponding variable
    *** If p2 <= p1, raise a ValueError with the message p2 must be
        greater than p1
    *** The only import you are allowed to use is from scipy import special
    Returns:
            the posterior probability that p is within the range
            [p1, p2] given x and n
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(p1, float)) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if (not isinstance(p2, float)) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    a = x + 1
    b = n - x + 1
    ac1 = special.btdtr(a, b, p1)
    ac2 = special.btdtr(a, b, p2)
    return ac2 - ac1
