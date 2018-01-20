import numpy as np

def zhao_xu_franti09(data, all_centers, all_labels) :
    maxK = len(all_centers)
    SSW = np.zeros((maxK-1))
    SSB = np.zeros((maxK-1))
    WB = np.zeros((maxK-1))

    for k in range(1, maxK) :
        centers = all_centers[k]
        labels = all_labels[k]
        for iter1 in range(k+1) :
            SSW[k-1] += np.sum( ( np.sum( ( data[labels==iter1,:]\
                - centers[iter1,:] )**2, 1 ) )**0.5 )
        SSW[k-1] /= data.shape[0]
        datamean = np.mean(data,0)
        for iter1 in range(k+1) :
            SSB[k-1] += ( ( np.sum( ( centers[iter1,:] - datamean )**2 )\
                )**0.5 ) * np.sum(labels==iter1)
        SSB[k-1] /= data.shape[0]
        WB[k-1] = (k+1)*SSW[k-1] / SSB[k-1]
    return np.argmin(WB)+2, WB
