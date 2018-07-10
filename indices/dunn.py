# Paper Source: J. C. Dunn,
# A Fuzzy Relative of the ISODATA process and its Use in Detecting
# Compact Well- Separated Clusters, Journal of Cybernetics 3 (3) (1973) 32–57.

# Min number of clusters = 2
# The time complexity is O(nk), where n is the number of data points
# and k is the number of clusters.
# To estimate the number of clusters: Use max of DUNN


import numpy as np
from scipy.spatial.distance import cdist


def dunn(pairwise_distances, labels):
    #pairwise_distances = cdist(data, data)

    inter_center_dists = +np.inf
    intra_center_dists = 0
    for iter1 in range(len(np.unique(labels))):
        inter_center_dists = min(
            inter_center_dists,
            pairwise_distances[labels==iter1,:][:,labels!=iter1].min()
        )
        intra_center_dists = max(
            intra_center_dists,
            pairwise_distances[labels==iter1,:][:,labels==iter1].max()
        )
    return inter_center_dists / np.fmax(intra_center_dists, 1.0e-16)


if __name__ == '__main__':
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    pairwise_distances = cdist(data, data)
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = dunn(
            pairwise_distances, km1.labels_
        )
    est_k = index.argmax() + minK
    print('For Iris:\nSelected k =', est_k)

    import matplotlib.pyplot as plt
    fig, axisArray = plt.subplots(2,2)
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
    pairwise_distances = cdist(data, data)
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = dunn(
            pairwise_distances, km1.labels_
        )
    est_k = index.argmax() + minK
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
