import numpy as np
from scipy.spatial.distance import cdist

def rezaee10(data, all_centers, all_labels) :
    maxK = len(all_centers)

    sep = np.zeros((maxK-1))
    comp = np.zeros((maxK-1))
    for k in range(1, maxK) :
        centers = all_centers[k]
        labels = all_labels[k]

        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = mu.T
        mu = np.fmax(mu, 1.0e-16)

        for iter1 in range(k+1) :
            comp[k-1] += np.sum( ( mu[:,iter1]**2 ) *\
                ( np.linalg.norm(data - centers[iter1,:], axis=1)**2 ) )

        for iter1 in range(k) :
            for iter2 in range(iter1+1,k+1) :
                sep[k-1] += np.sum( np.fmin(mu[:,iter1], mu[:,iter2]) *\
                    - np.sum(mu*np.log(mu), 1) )
        sep[k-1] *= ( 2 / (k * (k+1)) )

    comp = comp / np.amax(comp)
    sep = sep / np.amax(sep)

    return np.argmin(sep+comp)+2, (sep,comp)
