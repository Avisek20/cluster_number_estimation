import numpy as np
from scipy.spatial.distance import cdist

def bensaid_hall_bezdek(data, all_centers, all_labels, m = 2) :
    maxK = np.int(np.round(np.sqrt(data.shape[0])))

    VB = np.zeros((maxK-1))
    for k in range(2, maxK+1) :
        centers = all_centers[k-1]
        labels = all_labels[k-1]

        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = mu.T
        mu = np.fmax(mu, 1.0e-16)

        for iter1 in range(k) :
            VB[k-2] += np.sum((mu[:,iter1]**m) * (np.linalg.norm(data-centers[iter1,:], axis=1)**2))\
                / (np.sum(mu[:,iter1]) * np.sum(np.linalg.norm(centers-centers[iter1,:], axis=1)**2))
    return np.argmax((VB[2:]-VB[1:-1])-(VB[1:-1]-VB[0:-2]))+3, VB
