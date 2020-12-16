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
                raise ValueError("the message p must be greater\
                                than 0 and less than 1")
            self.p = p
            self.n = n
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data)/len(data)
            s = 0
            for x in data:
                s += (x - mean)**2
            std = (s/len(data))**0.5
            v = s/len(data)
            n = mean**2/(mean-v)
            self.n = round(n)
            self.p = mean/n