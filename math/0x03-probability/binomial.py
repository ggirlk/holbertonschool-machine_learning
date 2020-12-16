#!/usr/bin/env python3
""" class Binomial """


class Binomial():
    """ Represents a Binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ constractor """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
            self.n = int(n)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data)/len(data)
            s = 0
            for x in data:
                s += (x - mean)**2
            v = s/len(data)
            n = round((mean**2)/(mean-v))
            self.n = int(n)
            self.p = mean/n

    def fact(self, k):
        """ calculate factorial """
        fact = 1
        for i in range(1, k+1):
            fact = fact * i
        return fact

    def pmf(self, k):
        """
            Calculates the value of the PMF
            for a given number of “successes”
        """
        if k < 0:
            return 0
        try:
            k = int(k)
            nk = self.fact(self.n)/(self.fact(k)*self.fact(self.n - k))
            return nk * self.p**k * (1-self.p)**(self.n-k)
        except Exception:
            return 0

    def cdf(self, k):
        """
            Calculates the value of the CDF
            for a given number of “successes”
        """
        if k < 0:
            return 0
        try:
            k = int(k)
            cdf = 0
            for i in range(0, k+1):
                cdf += self.pmf(i)
            return cdf
        except Exception:
            return 0
