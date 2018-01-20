import numpy as np
from scipy.spatial.distance import cdist

def sillhouette(pairwise_distances, labels) :
    mu_in = np.zeros((labels.shape[0]))
    K = np.int(labels.max()) + 1
    if K == 1 :
        return np.zeros((labels.shape[0]))
    for iter1 in range(K) :
        mu_in[labels==iter1] = np.sum(pairwise_distances[labels==iter1,:][:,labels==iter1], 1) \
                 / np.fmax(np.sum(labels==iter1) - 1, 1.0e-16)
    mu_out_min = np.ones((labels.shape[0]))*np.inf
    for iter1 in range(K) :
        for iter2 in range(K) :
            if iter1 != iter2 :
                mu_out_min[labels==iter1] = np.fmin(np.sum(\
                    pairwise_distances[labels==iter1,:][:,labels==iter2], 1) \
                    / np.fmax(np.sum(labels==iter2), 1.0e-16), mu_out_min[labels==iter1])
    return ( mu_out_min - mu_in ) / ( np.fmax(mu_out_min, mu_in) )

def slope_statistic(data, all_centers, all_labels) :
	pairwise_distances = cdist(data, data)
	slope = np.zeros((len(all_centers)))
	for iter1 in range(len(all_centers)) :
		centers = all_centers[iter1]
		labels = all_labels[iter1]
		sil = sillhouette(pairwise_distances, labels)
		slope[iter1] = np.sum(sil)/data.shape[0]
	# Returns two estimates : Slope statistic, and largest average sillhouette
	return (np.argmax(- (slope[1:] - slope[0:-1]) * np.power(slope[0:-1],\
		data.shape[1]))+1, np.argmax(slope)+1), slope
