import numpy as np
from scipy.spatial.distance import cdist

def classification_entropy81(data, all_centers, all_labels) :
    maxK = len(all_centers)
    class_entropy = np.zeros((maxK))

    for k in range(maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = np.fmax(mu, np.finfo(np.float64).eps)
        class_entropy[k] = -np.sum(mu*np.log(mu))/data.shape[0]

    return np.argmin(class_entropy[1:])+2, class_entropy
