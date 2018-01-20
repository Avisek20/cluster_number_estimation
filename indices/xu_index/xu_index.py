import numpy as np
from scipy.spatial.distance import cdist

def xu_index(data, all_centers, all_labels) :
	xu = np.zeros((len(all_centers)-1))
	for iter2 in range(1,len(all_centers)) :
		labels = all_labels[iter2]
		k = int(labels.max())+1
		pairwise_sqeuclidean = cdist(data, data, metric='sqeuclidean')

		SSW = 0
		for iter1 in range(np.int(labels.max())+1) :
			SSW += np.sum(pairwise_sqeuclidean[labels==iter1,:][:,labels==iter1])
		SSW *= 0.5

		xu[iter2-1] = data.shape[1] * np.log( ( SSW \
			/ ( data.shape[1] * (data.shape[0]**2) ) )**0.5 ) + np.log(k)

	return np.argmin(xu[1:]-xu[0:-1])+3, xu
