import os
import numpy as np
from numpy.random import normal
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def generate_well_separated_clusters(num_clusters, cluster_size, data_dim) :
    # The vector of cluster centers for the synthetic data
    mean_vector = np.zeros((num_clusters, data_dim))
    # Generate first cluster center
    mean_vector[0,:] = \
        np.random.uniform(low=0,high=10*num_clusters,size=(1,data_dim))
    # Generate first cluster
    data_set = np.hstack((\
        normal(loc=mean_vector[0,:], scale=1, \
        size=(cluster_size, data_dim)), np.zeros((cluster_size, 1)) ))
    for iter1 in range(num_clusters-1) :
        # Generate next center at distance > 10 from all other centers
        flag = 1
        while flag :
            # Generate next center
            mean_vector[iter1+1,:] = \
            np.random.uniform(low=0,high=10*num_clusters,size=(1,data_dim))
            # Calculate distance to previous centers
            min_dist = cdist(mean_vector[0:iter1+1+1,:], \
            mean_vector[0:iter1+1+1,:])
            np.fill_diagonal(min_dist, np.inf)
            # Find minimum distance, if less than 10, loop again to
            # regenerate the center
            if min_dist.min() > 10 :
                flag = 0
        # Generate the next cluster
        data_set = np.vstack((data_set, \
            np.hstack((\
                normal(loc=mean_vector[iter1+1,:], scale=1, \
                size=(cluster_size, data_dim)),\
                    np.ones((cluster_size, 1))+iter1 \
                )),
            ))
    return data_set


if __name__ == '__main__' :

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Directory to save data, and which function to run
    # 1. data is well-separated, run all methods
    directory = 'data_well_separated'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # List of possible (i) number of clusters (ii) data dimensions
    list_num_clusters = [2, 10, 20, 35, 50]
    list_data_dim = [2, 10, 20, 35, 50]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if not os.path.exists(directory):
        os.makedirs(directory)
    for iter1 in range(30) :
        # Set up a directory to write the generated data, and the info file
        if not os.path.exists(directory+'/'+directory+'_'+str(iter1)):
            os.makedirs(directory+'/'+directory+'_'+str(iter1))
        fw = open(directory+'/'+directory+'_'+str(iter1)+'_info.txt', 'w')
        fw.write('idx\tcluster_num\tdim\n')
        count = 0
        for num_clusters in list_num_clusters:
            for data_dim in list_data_dim:
                print('Set:', iter1, 'No. of clusters:', num_clusters, \
                    'Data dim:', data_dim)
                fw.write(str(iter1)+'\t'+str(num_clusters)+'\t'+str(data_dim)+'\n')
                # Get generated data set
                data = \
                    generate_well_separated_clusters(num_clusters=num_clusters, \
                    cluster_size=100, data_dim=data_dim)
                print(data.shape)
                # Save data set
                np.savetxt(directory+'/'+directory+'_'+str(iter1)+ \
                    '/data_'+str(count)+'.txt', data)
                # Plot data set. Project to 2D using PCA if num of dimensions
                # is not 2
                '''
                if data.shape[1] == 2+1 :
                    plt.scatter(data[:,0], data[:,1], marker='x', c='b')
                    plot_axis_range(data[:,0:-1])
                else :
                    proj_data = PCA(n_components=2).fit_transform(data)
                    plt.scatter(proj_data[:,0], proj_data[:,1], marker='x', c='b')
                    plot_axis_range(proj_data)
                plt.savefig(directory+'/'+directory+'_'+str(iter1)+ \
                    '/data_'+str(count)+'.png', dpi=60)
                plt.close()
                '''
                count += 1
        fw.close()
