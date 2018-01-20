import os
import numpy as np
from numpy.random import normal
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from plot_axis_range import plot_axis_range
# high:
# if min_dist.min() >= 2 and min_dist.min() < 3  :

def generate_high_overlap(num_clusters, cluster_size, data_dim) :
    # When the number of clusters is >= 2, then create overlaps with
    # a probability
    overlap_probability = 0.5
    # The vector of cluster centers for the synthetic data
    mean_vector = np.zeros((num_clusters, data_dim))
    # Generate first cluster center
    mean_vector[0,:] = \
        np.random.uniform(low=0, high=10*num_clusters, size=(1,data_dim))
    # Generate first cluster
    data_set = np.hstack((\
        normal(loc=mean_vector[0,:], scale=1, \
        size=(cluster_size, data_dim)), \
        np.zeros((cluster_size, 1)) \
        ))
    num_well_separated_clusters = 1
    for iter1 in range(num_clusters-1) :
        # To overlap or not to overlap
        rand = np.random.rand()
        flag = 1
        while flag :
            if rand <= overlap_probability:
                # Pick a center
                which_center = np.random.randint(iter1+1)
                # Add noise to get a new center
                noise = np.random.uniform(low=-3,high=3,size=(1,data_dim))
                # New center
                mean_vector[iter1+1,:] = \
                    mean_vector[which_center,:] + noise
                # Calculate distance to the previous centers
                min_dist = cdist(mean_vector[0:iter1+1,:], \
                    mean_vector[iter1+1,:].reshape(1,data_dim))
                # For high overlap
                if min_dist.min() >= 2 and min_dist.min() <= 3:
                    flag = 0
                    break
            else:
                # Generate next center
                mean_vector[iter1+1,:] = \
                    np.random.uniform(low=0, high=10*num_clusters, \
                    size=(1,data_dim))
                # Calculate distance to previous centers
                min_dist = cdist(mean_vector[0:iter1+1,:], \
                    mean_vector[iter1+1,:].reshape(1,data_dim))
                # Find minimum distance, if less than 10, loop again to
                # regenerate the center (ensures that the clusters are well-separated)
                if min_dist.min() > 10:
                    num_well_separated_clusters += 1
                    flag = 0
                    break
        # Generate the next cluster
        data_set = np.vstack((data_set, \
            np.hstack((\
                normal(loc=mean_vector[iter1+1,:], scale=1, \
                size=(cluster_size, data_dim)),\
                    np.ones((cluster_size, 1))+iter1 \
                )),
            ))
    return data_set, num_well_separated_clusters

if __name__ == '__main__':

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    directory = 'data_high_overlap2'
    key_directory = 'key_high_overlap2'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # List of possible (i) number of clusters (ii) data dimensions
    list_num_clusters = [5, 10, 25, 50, 100]
    list_data_dim = [2]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(key_directory):
        os.makedirs(key_directory)
    for iter1 in range(100):
        # Set up a directory to write the generated data, and the info file
        if not os.path.exists(directory+'/'+directory+'_'+str(iter1)):
            os.makedirs(directory+'/'+directory+'_'+str(iter1))
        fw = open(directory+'/'+directory+'_'+str(iter1)+'_info.txt', 'w')
        fw.write('idx\tcluster_num\tdim\n')
        fw2 = open(key_directory+'/'+key_directory+'_'+str(iter1)+'.txt', 'w')
        count = 0
        for num_clusters in list_num_clusters:
            for data_dim in list_data_dim:
                print('Set:', iter1, 'No. of clusters:', num_clusters, \
                    'Data dim:', data_dim)
                fw.write(str(iter1)+'\t'+str(num_clusters)+'\t'+\
                    str(data_dim)+'\n')

                # Get generated data set
                data, num_well_separated_clusters = \
                    generate_high_overlap(num_clusters=num_clusters, \
                    cluster_size=200, data_dim=data_dim)
                print(data.shape)

                # Save data set
                np.savetxt(directory+'/'+directory+'_'+str(iter1)+ \
                    '/data_'+str(count)+'.txt', data)
                fw2.write(str(num_well_separated_clusters)+'\n')

                # Plot data set.
                plt.figure()
                plt.scatter(data[:,0], data[:,1], marker='x', c='k')
                plot_axis_range(data[:,0:-1])
                plt.savefig(directory+'/'+directory+'_'+str(iter1)+ \
                    '/data_'+str(count)+'.png', dpi=60)
                plt.close()

                count += 1
        fw.close()
        fw2.close()
