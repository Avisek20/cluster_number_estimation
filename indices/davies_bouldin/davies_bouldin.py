import numpy as np
from scipy.spatial.distance import cdist

def davies_bouldin(data, all_centers, all_labels) :
    # smaller is better
    DB = np.zeros((len(all_centers)-1))
    for iter3 in range(len(all_centers)-1) :
        centers = all_centers[iter3+1]
        labels = all_labels[iter3+1]
        k = np.int(labels.max()) + 1
        
        clust_sigma = np.zeros((k))
        for iter1 in range(k) :
            clust_sigma[iter1] = np.sqrt(np.sum( (np.linalg.norm(data[labels==iter1,:] \
                - centers[iter1,:], axis=1)**2), 0) / np.sum(labels==iter1) )

        center_dists = cdist(centers, centers)
        DB_cluster_pairs = np.zeros((k, k))
        for iter1 in range(k) :
            for iter2 in range(k) :
                if iter1 < iter2 :
                    DB_cluster_pairs[iter1, iter2] = ( clust_sigma[iter1] + clust_sigma[iter2] ) / center_dists[iter1,iter2]
                elif iter2 < iter1 :
                    DB_cluster_pairs[iter1, iter2] = DB_cluster_pairs[iter2, iter1]

        DB[iter3] = np.sum(np.amax(DB_cluster_pairs, 0)) / k

    return np.argmin(DB)+2, DB
