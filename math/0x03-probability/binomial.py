#!/usr/bin/env python3
""" class Binomial """


class Binomial():
    """ Represents a Binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ constractor """
        if data is None:
            if n <= 0:
                raise ValueError("the message n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("the message p must be greater than 0 and less than 1")
            self.p = p
            self.n = n
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.p = sum(data)*2/len(data)**2
            self.n = len(data)/2
