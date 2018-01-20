import numpy as np

def jump_method(data, all_centers, all_labels) :
	dk = np.zeros((len(all_centers)+1))
	dk[0] = 1.0e-16
	slope = np.zeros((len(all_centers)))
	for iter1 in range(1,len(all_centers)+1) :
		centers = all_centers[iter1-1]
		labels = all_labels[iter1-1]
		for iter2 in range(iter1) :
			dk[iter1] += np.sum((data[labels==iter2,:] - np.atleast_2d(centers)[iter2,:])**2)
		dk[iter1] /= data.shape[1]
	Jk = ((dk[1:])**(-data.shape[1]/2)) - (dk[0:-1]**(-data.shape[1]/2))
	return np.argmax(Jk)+1, Jk
