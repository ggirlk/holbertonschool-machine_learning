#!/usr/bin/env python3
""" doc """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if you should stop gradient descent early """
    test = opt_cost - cost > threshold
    if test:
        count = 0
    else:
        count += 1
    return (count >= patience, count)
