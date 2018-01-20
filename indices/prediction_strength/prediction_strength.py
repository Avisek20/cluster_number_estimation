import sys
import random
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
#import matplotlib.pyplot as plt


def k_fold (data,shuffled) :
	testSample = list()
	trainingSample = list()
	for i in range(data.shape[0]) :
		if i in shuffled :
			testSample.append(data[i])
		else:
			trainingSample.append(data[i])
	return np.array(trainingSample), np.array(testSample)

def closest_center (point, centroids) :
    '''
	min_index = -1
	min_distance = np.inf
	for i in range(len(centroids)) :
		center = centroids[i]
		d = sum((point - center)**2)
		if d < min_distance :
			min_index = i
			min_distance = d
	return min_index
    '''
    return np.argmin(cdist(np.atleast_2d(point), centroids), axis=1)

def calculate_prediction_strength (test_set, test_labels, training_centers, k) :
	clusterLength = test_labels.tolist().count(k)
	if clusterLength <= 1 :
		return np.inf
	else:
		count = 0.
		for i in range(len(test_labels)-1) :
			for j in range(i+1, len(test_labels)) :
				if test_labels[i] == test_labels[j] == k :
					p1 = test_set[i]
					p2 = test_set[j]
					if (closest_center(p1, training_centers) == closest_center(p2, training_centers)) :
						count += 1
		# Return the proportion of pairs that stayed in the same cluster.
		#print count,(clusterLength * (clusterLength - 1) / 2.), clusterLength
		count = count / (clusterLength * (clusterLength - 1) / 2)
	return count

def prediction_strength_of_cluster (test_set, test_labels, training_centers, k) :
	prediction_strengths = [calculate_prediction_strength(test_set, test_labels, training_centers, i) for i in range(k)]
	return min(prediction_strengths)


def run_prediction_strength(matrix) :

    #if __name__ == '__main__' :
    #start = time.time()
    '''
    matrix = np.vstack(( np.vstack(( \
        np.random.normal(loc = [0,0], scale=1, size=(200,2)), \
        np.random.normal(loc = [10,0], scale=1, size=(200,2)) \
        )),
        np.random.normal(loc = [5,10], scale=1, size=(200,2))
        ))
    '''

    population = range(matrix.shape[0])
    testSetLength = int(matrix.shape[0]/10)
    choice = random.sample(population, testSetLength)

    maxK = int(matrix.shape[0]**0.5)
    maxTrials = 30
    prediction_strengths = np.zeros((maxTrials, maxK))
    trainingSet, testingSet = k_fold(matrix, choice)
    for trial in range(maxTrials):
    	#trainingSet, testingSet = k_fold (matrix,choice)
    	for k in range(1,maxK+1):
    		if k==1:
    			prediction_strengths[trial, k-1] = 1.
    		else:
    			testCluster = KMeans(n_clusters=k, max_iter=300, n_init=3, tol=1e-16).fit(testingSet)
    			trainingCluster = KMeans(n_clusters=k, max_iter=300, n_init=3, tol=1e-16).fit(trainingSet)

    			prediction_strengths[trial,k-1] = prediction_strength_of_cluster(testingSet,
    											 testCluster.labels_, trainingCluster.cluster_centers_, k)

    means = np.mean(prediction_strengths,0)
    stddevs = np.std(prediction_strengths,0)

    #We use 0.8 as our prediction strength threshold.
    #Find the largest number of clusters with a prediction strength greater than this threshold;
    #this forms our estimate of the number of clusters.

    #print(means)
    if max(means) > 0.8:
    	estimated = max([i for i,j in enumerate(means) if j > 0.8])+1
    else:
    	estimated = max(means)+1

    #print("The estimated number of clusters is ", estimated)
    print(estimated)
    #print(time.time()-start)
    #print range(1,maxK+1), means
    #plt.plot(range(1,maxK+1),means)
    #plt.show()
    return estimated, means
