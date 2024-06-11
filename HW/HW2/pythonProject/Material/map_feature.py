# -*- coding: utf-8 -*-

import numpy as np

def map_feature(x1, x2, degree = 6):
    '''
    Maps a two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    x1, x2, x1 ** 2, x2 ** 2, x1*x2, x1*x2 ** 2, etc...
    The inputs x1, x2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    X = np.ones(shape=(x1[:, 0].size, 1))
   
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            X = np.append(X, r, axis = 1)

    return X