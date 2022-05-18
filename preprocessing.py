import numpy as np


def none(train, test):
    return [train, test]

def minmax(train, test):
    min = np.min(train)
    max_min = np.max(train-min)
    return [(train-min)/max_min,
            (test-min)/max_min]

def normalize(train, test):
    mu = np.mean(train)
    sigma = np.std(train, axis=0)
    sigma[np.where(sigma==0)] = 1
    return [(train-mu)/sigma,
            (test-mu)/sigma]
