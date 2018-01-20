import os
import numpy as np
from numpy.random import normal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from plot_axis_range import plot_axis_range

if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~ #

    list_directories = ['data_diff_spread']

    # ~~~~~~~~~~~~~~~~~~~~~~~ #
    for directory in list_directories:
        cluster_results_directory = 'cluster_'+directory
        if not os.path.exists(cluster_results_directory):
            os.makedirs(cluster_results_directory)
        for iter1 in range(50):
            if not os.path.exists( \
                cluster_results_directory+'/'+directory+'_'+str(iter1) \
                ):
                os.makedirs(cluster_results_directory+'/'+directory+'_'+str(iter1))
            for iter2 in range(25):
                # Load the data
                data = np.loadtxt(directory+'/'+directory+'_'+str(iter1)+'/data_'+str(iter2)+'.txt')
                data = data[:,0:-1]
                maxK = int(np.ceil(data.shape[0]**0.5))

                for k in range(1, maxK+1) :
                    # Run k-Means
                    km1 = KMeans(n_clusters=k, max_iter=300, n_init=30,\
                    tol=1e-16).fit(data)
                    # Write centers
                    np.savetxt(cluster_results_directory+'/'+directory+'_'+str(iter1)+'/data_'+str(iter2)+'_k_'+str(k)+'_centers.txt', km1.cluster_centers_)
                    # Also write labels
                    np.savetxt(cluster_results_directory+'/'+directory+'_'+str(iter1)+'/data_'+str(iter2)+'_k_'+str(k)+'_labels.txt', km1.labels_)
