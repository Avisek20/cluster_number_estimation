import numpy as np
from scipy.spatial.distance import cdist

def inter_center_distance(centers) :
	dist = cdist(centers, centers, metric='sqeuclidean')
	np.fill_diagonal(dist, np.inf)
	return dist.min()

def LastLeap(all_centers, all_labels) :
	maxK = len(all_centers)

	min_d = np.zeros((maxK - 2 + 1))
	for iter1 in range(2, maxK+1) :
		min_d[iter1-2] = inter_center_distance(all_centers[iter1-1])

	k_est1 = np.argmax( (min_d[0:-1]-min_d[1:])/min_d[0:-1] ) + 2

	rest_of_the_data = min_d[k_est1-2+1:]
	if np.sum(min_d[k_est1-2]*0.5 < rest_of_the_data) > 0 :
		k_est1 = 1

	return k_est1, min_d

def LastMajorLeap(all_centers, all_labels) :
	maxK = len(all_centers)

	min_d = np.zeros((maxK - 2 + 1))
	for iter1 in range(2, maxK+1) :
		min_d[iter1-2] = inter_center_distance(all_centers[iter1-1])

	# Estimate k
	for iter1 in range(min_d.shape[0]-2, 0-1, -1) :
	    if min_d[iter1]*0.5 > np.amax(min_d[iter1+1:]) :
	        k_est1 = iter1+2
	        break

	return k_est1, min_d
