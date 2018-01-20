import numpy as np
from scipy.spatial.distance import cdist

def partition_coefficient74(data, all_centers, all_labels) :

    maxK = len(all_centers)
    part_coeff = np.zeros((maxK-1))
    for k in range(1, maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        dist = cdist(data, centers).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = (mu ** 2).T
        part_coeff[k-1] = np.sum(mu)/data.shape[0]
    return np.argmax(part_coeff)+2, part_coeff
