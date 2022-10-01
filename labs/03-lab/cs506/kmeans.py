from collections import defaultdict
from math import inf
from symbol import return_stmt
from matplotlib.artist import get
import numpy as np
import random
import csv

from cs506.sim import euclidean_dist

'''
Attempted to do this for a day or so.

I think I understand the material regarding kmeans and k++ means.

However, the difficulty in this lab is not understanding the material.
Rather that it is understanding how all of the functions that are pre-implemented
and not pre-implemented are tied together. 

Like what is a clustering? Is it an array of arrays, a dictionary where the keys correspond to the cluster, etc.

Also, what does the comment for generate_k_pp mean?
     Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
I pick a random set of k points from the data_set where the points are picked with a
probability proportional to their distance as per kmeans pp from what? The center?

generate_k_pp sounds like k++ mean, but I have no clue how this function is used lol.
Is this the function that is supposed to choose the next centroids based on probability, but then why is there
the get centroids function?
'''

def get_centroid(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    ls = []
    if len(points) >= 1:
        for i in range(len(points[0])):
            ls.append(0)

        for point in points:
            for i in range(len(point)):
                ls[i] += point[i]

        for j in range(len(ls)):
            ls[i] /= len(ls)
    
    return ls

        


def get_centroids(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the centroid for each of the assigned groups.
    Return `k` centroids in a list
    """

    dictionary = {}

    size = len(dataset[0])

    center = []

    for i in range(size):
        center.append(0)

    centerlsls = []

    centerlsls.append(center)

    maxIndex = max(np.unique(np.array(assignments)))

    for index in np.unique(np.array(assignments)):
        dictionary.update({index: []})


    for i in range(len(assignments)):
        ls = dictionary.get(assignments[i])
        ls.append(dataset[i])
        dictionary.update({assignments[i]: ls})
    
    centroidList = []

    for index in range(maxIndex + 1):
        if not (index in dictionary.keys()):
            dictionary.update({index: centerlsls})
        

    for x in dictionary:
        centroidList.append(get_centroid(dictionary[x]))
    
    return centroidList

def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return euclidean_dist(a,b)

def distance_squared(a, b):
    return distance(a,b) * distance(a,b)


def cost_function(clustering):
    cost = 0
    centroid = get_centroid(clustering[0])
    for point in clustering[0]:
        cost += distance_squared(point, centroid)
    return cost


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    return random.sample(dataset, k)


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    #Not sure what per kmeans pp means
    raise NotImplementedError


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = get_centroids(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
