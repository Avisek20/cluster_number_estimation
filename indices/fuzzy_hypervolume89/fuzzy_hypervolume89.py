import numpy as np
from scipy.spatial.distance import cdist

def fuzzy_hypervolume89(data, all_centers, all_labels, m=2) :
    maxK = len(all_centers)
    FHV = np.zeros((maxK))
    for k in range(maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = (mu ** m).T
        for iter1 in range(k+1) :
            FHV[k] += np.sqrt(np.absolute(np.sum(mu[:,iter1] * \
                np.linalg.norm(data - np.atleast_2d(centers)[iter1,:], \
                axis=1)**2) / np.sum(mu[:,iter1])))

    return np.argmin(FHV)+1, FHV
