import numpy as np
from scipy.spatial.distance import cdist

def dunn73_dist(pairwise_distance, labels) :
    k = np.int(labels.max()) + 1
    dist_mat = np.zeros((k,k))
    for iter1 in range(k) :
        for iter2 in range(k) :
            if iter2 == iter1 :
                continue
            elif iter2 > iter1 :
                dist_mat[iter1,iter2] = np.amin(pairwise_distance[labels==iter1,:][:,labels==iter2])
            else :
                dist_mat[iter1,iter2] = dist_mat[iter2,iter1]
    np.fill_diagonal(dist_mat,np.inf)
    return np.amin(dist_mat)

def dunn73_diam(pairwise_distance, labels) :
    k = np.int(labels.max()) + 1
    diam_mat = np.zeros((k))
    for iter1 in range(k) :
        diam_mat[iter1] = np.argmax(pairwise_distance[labels==iter1,:][:,labels==iter1])
    return np.amax(diam_mat)

def dunn73(data, all_centers, all_labels) :
    maxK = len(all_centers)
    pairwise_distance = cdist(data, data)
    dunn73 = np.zeros((maxK-1))

    for k in range(maxK-1) :
        centers = all_centers[k+1]
        labels = all_labels[k+1]
        dunn73[k] = dunn73_dist(pairwise_distance, labels) / np.fmax(dunn73_diam(pairwise_distance, labels), 1.0e-16)
    return np.argmax(dunn73)+2, dunn73
