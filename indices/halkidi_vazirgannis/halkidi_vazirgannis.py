import numpy as np

def cluster_density(data, center, labels, clustVar) :
    return np.sum(np.sum((data[labels,:]-center)**2, 1) <= clustVar)

def halkidi_vazirgannis(data, all_centers, all_labels) :
    maxK = len(all_centers)
    scat = np.zeros((maxK))
    dens_bw = np.zeros((maxK))

    for k in range(maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        clustVar = np.zeros((k+1))
        densVar = np.zeros((k+1))
        for iter1 in range(k+1) :
            clustVar[iter1] = np.sum((data[labels==iter1,:] - np.atleast_2d(centers)[iter1,:])**2)/np.sum(labels==iter1)
            densVar[iter1] = cluster_density(data, np.atleast_2d(centers)[iter1,:], labels==iter1, clustVar[iter1])
            scat[k] += clustVar[iter1]
        scat[k] /= ((k+1) * (np.sum((data - np.mean(data,axis=0))**2) \
            / data.shape[0]))
        for iter1 in range(k+1) :
            for iter2 in range(k+1) :
                if iter2 == iter1 :
                    continue
                dens_bw[k] += cluster_density(data,\
                    (centers[iter1,:]+centers[iter2,:])/2, \
                    (labels==iter1)+(labels==iter2),\
                    (clustVar[iter1]+clustVar[iter2])/2)\
                    / max(densVar[iter1],densVar[iter2])
        dens_bw[k] /= ((k+1)*np.fmax(k,1.0e-16))

    return np.argmin(scat + dens_bw)+1, (scat, dens_bw)
