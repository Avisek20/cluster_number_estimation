import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def krzanowski_lai(data, all_centers, all_labels) :
	maxK = len(all_centers)
	
	Wk = np.zeros((maxK))
	for k in range(1,maxK+1) :
		centers = all_centers[k-1]
		labels = all_labels[k-1]
		for iter1 in range(k) :
			Wk[k-1] += np.sum((data[labels==iter1,:] - centers[iter1,:])**2)
	
	d = data.shape[1]
	DIFF = np.zeros((maxK-1))
	for k in range(2,maxK+1) :
		DIFF[k-2] = (((k-1)**(2/d)) * Wk[k-2]) - ((k**(2/d)) * Wk[k-1])

	KLI = np.zeros((maxK-2))
	for k in range(2,maxK) :
		KLI[k-2] = np.absolute(DIFF[k-2] / DIFF[k-1])
	
	return np.argmax(KLI) + 2, KLI

if __name__ == '__main__' :
	from sklearn.cluster import KMeans
	data = \
		np.vstack(( \
		np.vstack(( \
			np.random.normal(loc = [0,0], scale=1, size=(200,2)), \
			np.random.normal(loc = [10,0], scale=1, size=(200,2)) \
		)),
			np.random.normal(loc = [5,10], scale=1, size=(200,2))
		))

	all_centers = []
	all_labels = []
	maxK = int((data.shape[0])**0.5)
	for iter1 in range(1,maxK+1) :
		km1 = KMeans(n_clusters=iter1, max_iter=300, n_init=10, tol=1e-16).fit(data)
		all_centers.append(km1.cluster_centers_)
		all_labels.append(km1.labels_)
	k, KL = krzanowski_lai(data, all_centers, all_labels)
	print(k)
	print(KL)
	plt.scatter(data[:,0], data[:,1], marker='x', c='k')
	plt.show()
	