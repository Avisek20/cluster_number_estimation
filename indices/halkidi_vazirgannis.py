# Paper Source: M. Halkidi, M. Vazirgiannis 2001.
# Clustering Validity Assessment: Finding the optimal partitioning
# of a data set. Proceedings of the
# 2001 IEEE International Conference on Data Mining (ICDM 2001),
# pp. 187–194.

# Min number of clusters = 1
# The time complexity is O(nk^2), where n is the number of data points
# and k is the number of clusters.
# To estimate the number of clusters: Use min of S_Dbw


import numpy as np
from scipy.spatial.distance import cdist


def halkidi_vazirgannis(data, centers, labels):
    var_clust = np.zeros((centers.shape[0], data.shape[1]))
    for i in range(centers.shape[0]):
        var_clust[i] = ((data[labels==i] - centers[i]) ** 2).sum(axis=0) / np.fmax((labels==i).sum(axis=0), np.finfo(float).eps)

    data_clust = (((data - data.mean(axis=0)) ** 2).sum(axis=0)) / data.shape[0]

    scat = np.linalg.norm(var_clust, axis=1).sum() / (centers.shape[0] * np.linalg.norm(data_clust))

    avg_std = ((np.linalg.norm(var_clust, axis=1).sum() ** 0.5)
        / centers.shape[0])

    dens = np.zeros((centers.shape[0], centers.shape[0]))
    for iter1 in range(centers.shape[0]):
        for iter2 in range(centers.shape[0]):
            if iter1 == iter2:
                dens[iter1,iter2] = (
                    cdist(data[labels==iter1], centers[iter1][None,:])
                    <= avg_std
                ).sum()
            else:
                dens[iter1,iter2] = (
                    cdist(
                        data[np.logical_or(labels==iter1, labels==iter2)], ((centers[iter1]+centers[iter2]) * 0.5)[None,:]
                    )
                    <= avg_std
                ).sum()
    for iter1 in range(centers.shape[0]):
        for iter2 in range(centers.shape[0]):
            if iter1 != iter2:
                dens[iter1,iter2] = (
                    dens[iter1,iter2] / np.fmax(
                        max(dens[iter1,iter1], dens[iter2,iter2])
                        , np.finfo(float).eps
                    )
                )
    np.fill_diagonal(dens, 0)
    dens_bw = dens.sum() / (centers.shape[0] * (centers.shape[0] - 1))

    return scat + dens_bw


if __name__ == '__main__':
    import time
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = halkidi_vazirgannis(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmin() + minK
    print('For Iris:\nSelected k =', est_k)

    import matplotlib.pyplot as plt
    fig, axisArray = plt.subplots(2,4)
    axisArray[0,0].scatter(data[:,2], data[:,3], marker='x', c='gray')
    axisArray[1,0].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,0].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,0].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,0].set_xticks(np.arange(index.shape[0])+minK)

    data = np.vstack(( np.vstack(( np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([20,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([10,10]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = halkidi_vazirgannis(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmin() + minK
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)


    data = np.vstack(( np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([20,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = halkidi_vazirgannis(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmin() + minK
    print('For 4 Gaussians:\nSelected k =', est_k)
    axisArray[0,2].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,2].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,2].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,2].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,2].set_xticks(np.arange(index.shape[0])+minK)


    data = np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = halkidi_vazirgannis(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmin() + minK
    print('For 3 Gaussians:\nSelected k =', est_k)
    axisArray[0,3].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,3].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,3].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,3].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,3].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
