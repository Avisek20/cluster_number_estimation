import numpy as np
from scipy.spatial.distance import cdist

def modified_partition_coefficient96(data, all_centers, all_labels) :
	maxK = len(all_centers)
	mod_part_coeff = np.zeros((maxK))

	for k in range(maxK) :
		centers = all_centers[k]
		labels = all_labels[k]
		dist = cdist(data, np.atleast_2d(centers)).T
		dist = np.fmax(dist, np.finfo(np.float64).eps)
		mu = dist ** (-2.0)
		mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
		mu = (mu ** 2).T
		mod_part_coeff[k] = 1 - ((k+1)/np.fmax(k,1.0e-16)) *\
		 	( 1 - (np.sum(mu)/data.shape[0]) )

	return np.argmax(mod_part_coeff[1:])+2, mod_part_coeff
	
