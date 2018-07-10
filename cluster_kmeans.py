import os
import numpy as np
from sklearn.cluster import KMeans


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~ #
    data_dir_path = '/home/arka/avisek/re_experiment2/datasets/'
    directory = [#'data_well_separated', 'data_diff_spread',
        'data_slightly_diff_points', 'data_very_diff_points',
        'data_slight_overlap2', 'data_high_overlap2']
    n_sets = [#30, 30,
    30, 30, 30, 30]
    n_datasets_in_sets = [#25, 25,
    25, 25, 5, 5]
    # ~~~~~~~~~~~~~~~~~~~~~~~ #
    for dir, ns, nds in zip(directory, n_sets, n_datasets_in_sets):
        cluster_results_directory = 'kmeans_cluster_'+dir
        if not os.path.exists(cluster_results_directory):
            os.makedirs(cluster_results_directory)

        for iter1 in range(ns):
            if not os.path.exists( \
                cluster_results_directory+'/'+dir+'_'+str(iter1) \
                ):
                os.makedirs(cluster_results_directory+'/'+dir+'_'+str(iter1))
            for iter2 in range(nds):
                # Load the data
                data = np.loadtxt(data_dir_path+dir+'/'+dir+'_'+str(iter1)+'/data_'+str(iter2)+'.txt')
                print(dir+'_'+str(iter1)+'/data_'+str(iter2)+'.txt', data.shape)
                data = data[:,0:-1]
                maxK = int(np.ceil(data.shape[0]**0.5))
                #'''
                for k in range(1, maxK+1):
                    # Run k-Means
                    km1 = KMeans(n_clusters=k, max_iter=300, n_init=10, tol=1e-9).fit(data)
                    # Write centers
                    np.savetxt(cluster_results_directory+'/'+dir+'_'+str(iter1)+'/data_'+str(iter2)+'_k_'+str(k)+'_centers.txt', km1.cluster_centers_, fmt='%.6f')
                    # Also write labels
                    np.savetxt(cluster_results_directory+'/'+dir+'_'+str(iter1)+'/data_'+str(iter2)+'_k_'+str(k)+'_labels.txt', km1.labels_, fmt='%d')
                #'''
