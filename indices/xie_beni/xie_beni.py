import numpy as np
from scipy.spatial.distance import cdist

def xie_beni(data, all_centers, all_labels) :
    maxK = len(all_centers)
    XB = np.zeros((maxK-1))
    for k in range(1, maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        for iter1 in range(k+1) :
            XB[k-1] += np.sum((data[labels==iter1,:] - centers[iter1,:])**2)
        XB[k-1] /= data.shape[0]
        d1 = np.triu(cdist(centers, centers, metric='sqeuclidean'), k=1)
        d1[d1==0] = np.inf
        XB[k-1] /= np.fmax(np.argmin(d1), 1.0e-16)

    return np.argmin(XB[1:]-XB[0:-1])+3, XB
