import numpy as np

def intra_cluster_scatter_matrix(data, centers, labels) :
    k = np.int(labels.max()) + 1
    SW = np.zeros((data.shape[1], data.shape[1]))
    for iter1 in range(k) :
        SW += np.dot((data[labels==iter1,:] - centers[iter1,:]).T, \
                    (data[labels==iter1,:] - centers[iter1,:]))
    return SW

def hartigan85(data, all_centers, all_labels) :
    k_range = len(all_centers)
    Hart = np.zeros((k_range-1))
    Wk = np.zeros((k_range, data.shape[1], data.shape[1]))
    for iter1 in range(k_range) :
        centers = all_centers[iter1]
        labels = all_labels[iter1]
        Wk[iter1,:,:] = intra_cluster_scatter_matrix(data, \
            np.atleast_2d(centers), labels)
        if iter1 > 0 :
            Hart[iter1-1] = ( ( np.trace(Wk[iter1,:,:]) \
                / np.trace(Wk[iter1-1,:,:]) ) - 1 ) * (data.shape[0] - iter1)

    return np.argmin(Hart)+2, Hart
