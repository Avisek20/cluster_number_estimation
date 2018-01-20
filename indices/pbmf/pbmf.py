import numpy as np
from scipy.spatial.distance import cdist

def pbmf(data, all_centers, all_labels, m=2, p=2) :
    maxK = len(all_centers)
    PBMF = np.zeros((maxK-1))
    E1 = np.sum(np.linalg.norm(data - np.mean(data, axis=0), axis=1))
    for k in range(1, maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = (mu ** m).T
        for iter1 in range(k+1) :
            mu[:,iter1] *= np.linalg.norm(data-centers[iter1,:], axis=1)
        Dk = np.triu(cdist(centers, centers), k=1)
        Dk[Dk==0] = np.inf
        Dk = np.amin(Dk)
        PBMF[k-1] = ((E1*Dk)/(np.sum(mu)*(k+1)))**p
    return np.argmax(PBMF)+2, PBMF
