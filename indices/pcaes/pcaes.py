import numpy as np
from scipy.spatial.distance import cdist

def pcaes(data, all_centers, all_labels) :
	maxK = len(all_centers)
	PCAES = np.zeros((maxK-1))

	for k in range(1, maxK) :
		centers = all_centers[k]
		labels = all_labels[k]
		dist = cdist(data, np.atleast_2d(centers)).T
		dist = np.fmax(dist, np.finfo(np.float64).eps)
		mu = dist ** (-2.0)
		mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
		mu = (mu**2).T

		mu_M = np.amin(np.sum(mu,0))
		beta_T = np.sum(np.linalg.norm(centers-np.mean(centers,axis=0),\
			axis=1)**2) / (k+1)
		for iter1 in range(k+1) :
			cd1 = np.linalg.norm(centers - centers[iter1,:], axis=1)
			cd1[iter1] = np.inf
			PCAES[k-1] += (np.sum(mu[:,iter1]) / mu_M) - \
				np.exp( - np.amin(cd1)**2 / beta_T )

	#print(PCAES)
	#print(np.argmax(PCAES[1:] - PCAES[0:-1])+2)

	return np.argmin(PCAES)+2, PCAES
