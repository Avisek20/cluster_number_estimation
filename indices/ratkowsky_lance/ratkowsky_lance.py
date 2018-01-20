import numpy as np

def ratkowsky_lance(data, all_centers, all_labels) :
    RL = np.zeros((len(all_centers)-1))
    for iter2 in range(1,len(all_centers)) :
        labels = all_labels[iter2]
        k = int(labels.max())+1
        num_points = np.zeros((k))
        for iter1 in range(k) :
            num_points[iter1] = np.sum(labels==iter1)

        dim_SSB = np.sum( ((centers - np.mean(data, axis=0))**2) \
            * np.atleast_2d(num_points).T , axis=0)

        dim_TSS = np.sum(( data - np.mean(data, axis=0) )**2 , axis=0)

        RL[iter2-1] = np.mean(np.sqrt(dim_SSB / dim_TSS)) / k
    return np.argmin(RL)+2
