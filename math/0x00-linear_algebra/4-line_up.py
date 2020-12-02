#!/usr/bin/env python3
""" add two arrays element-wise """



def add_arrays(arr1, arr2):
    """ add """
    try:
        new = []
        for i in range(0, len(arr1)):
            sm = arr1[i] + arr2[i]
            new.append(sm)
        return(new)
    except:
        return None
