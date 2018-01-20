import numpy as np

def cluster_density2(centers) :
    k = centers.shape[0]
    Dis = 0
    Dmax = 0
    Dmin = np.inf
    for iter1 in range(k) :
        for iter2 in range(k) :
            if iter2 == iter1 :
                continue
            dist1 = np.sum((centers[iter1,:]-centers[iter2,:])**2)
            if dist1 > Dmax :
                Dmax = dist1
            if dist1 < Dmin :
                Dmin = dist1
            Dis += 1.0/np.fmax(dist1, 1.0e-16)
    Dis *= (Dmax/Dmin)
    return Dis

def rezaee_lelieveldt_reiber98(data, all_centers, all_labels) :
    maxK = len(all_centers)
    scat = np.zeros((maxK-1))
    dis_bw = np.zeros((maxK-1))

    for k in range(1, maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        for iter1 in range(k+1) :
            scat[k-1] += np.sum((data[labels==iter1,:] - centers[iter1,:])**2)\
                / np.sum(labels==iter1)
        scat[k-1] /= ( (k+1) * ( np.sum((data - np.mean(data,axis=0))**2) \
			/ data.shape[0] ) )
        dis_bw[k-1] = cluster_density2(centers)

    return np.argmin(dis_bw[maxK-2]*scat + dis_bw)+2, (scat, dis_bw)
