# Paper Source: P. J. Rousseeuw, 1987.
# Silhouettes: a graphical aid to the interpretation
# and validation of cluster analysis.
# Journal of computational and applied mathematics, 20(1987) pp. 53-65.

# Min number of clusters = 2
# The time complexity is O(n^2), where n is the number of data points
# To estimate the number of clusters: Use max of silhouette


import numpy as np
from scipy.spatial.distance import cdist


def silhouette(pairwise_distances, labels) :
    k = len(np.unique(labels))
    a = np.zeros((pairwise_distances.shape[0]))
    for iter1 in range(k):
        denom = (labels==iter1).sum()
        if denom == 1:
            denom += np.finfo(float).eps
        a[labels==iter1] = (
            pairwise_distances[labels==iter1,:][:,labels==iter1]
        ).sum(axis=1) / denom
    b = np.zeros((pairwise_distances.shape[0])) + np.inf
    for iter1 in range(k):
        for iter2 in range(k):
            if iter1 == iter2:
                continue
            b[labels==iter1] = np.minimum(b[labels==iter1], (
                (
                    pairwise_distances[labels==iter1,:][:,labels==iter2]
                ).sum(axis=1) / np.fmax((labels==iter2).sum(), np.finfo(float).eps)
            ))
    s = (b - a) / np.maximum(b, a)
    return s.mean()


if __name__ == '__main__':
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    pairwise_distances = cdist(data, data)
    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = silhouette(
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

    pairwise_distances = cdist(data, data)
    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = silhouette(
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
