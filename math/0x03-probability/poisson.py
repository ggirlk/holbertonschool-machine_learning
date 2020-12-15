#!/usr/bin/env python3
""" class Poisson """


class Poisson():
    """ Represent a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ constractor """        
        if data == None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """
            Calculates the value of the PMF
            for a given number of “successes”
        """
        try:
            k = int(k)
            if k < 0:
                return 0
            lmda = self.lambtha
            fact = 1
            for i in range(1, k+1): 
                fact = fact * i
            return 2.7182818285**-lmda*lmda**k/fact
        except:
            return 0

    def cdf(self, k):
        """
            Calculates the value of the CDF
            for a given number of “successes”
        """
        try:
            k = int(k)
            if k < 0:
                return 0
            cdf = 0
            for i in range(0, k+1):
                cdf += self.pmf(i)
            return cdf
        except:
            return 0
