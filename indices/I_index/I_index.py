import numpy as np
from scipy.spatial.distance import cdist

def I_index(data, all_centers, all_labels, p=2) :
    maxK = len(all_centers)
    II = np.zeros((maxK-1))
    E1 = np.sum(np.linalg.norm(data - np.mean(data,axis=0),axis=1))
    for k in range(2, maxK+1) :
        centers = all_centers[k-1]
        labels = all_labels[k-1]
        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = mu.T
        for iter1 in range(k) :
            mu[:,iter1] *= np.linalg.norm(data-centers[iter1,:], axis=1)
        Dk = np.triu(cdist(centers, centers), k=1)
        Dk[Dk==0] = np.inf
        Dk = np.amin(Dk)
        II[k-2] = ((E1*Dk)/(np.sum(mu)*k))**p
    return np.argmax(II)+2, II
