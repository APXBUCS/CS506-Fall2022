from logging import raiseExceptions
import numpy as np

def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += abs(x[i] - y[i])
    return res

def jaccard_dist(x, y):
    if(len(x) == 0 and len(y) == 0):
        return 1
    intersection = [value for value in set(x) if value in set(y)]
    union = list(set(x) | set(y))
    return 1 - (len(intersection)) / (len(union))

def cosine_sim(x, y):
    if (len(x) == 0 or len(y) == 0):
        return 0
    if (len(x) != len(y)):
        return 0
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))
