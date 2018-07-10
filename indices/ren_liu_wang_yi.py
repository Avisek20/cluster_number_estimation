# Paper Source: Ren, M., Liu, P., Wang, Z. and Yi, J., 2016.
# A self-adaptive fuzzy c-means algorithm for
# determining the optimal number of clusters.
# Computational intelligence and neuroscience, 2016.

# Min number of clusters = 2
# The time complexity is O(nk), where n is the number of data points and k is the number of clusters.
# To estimate the number of clusters: Use min of RLWY


import numpy as np
from scipy.spatial.distance import cdist


def ren_liu_wang_yi(data, centers, labels, m=2) :
    dist = cdist(centers, data, metric='sqeuclidean')
    u = 1 / np.fmax(dist, np.finfo(float).eps)
    um = (u / u.sum(axis=0)) ** m

    return (
        (
            ((um * dist).sum(axis=1) / um.sum(axis=1))
            + (
                cdist(centers.mean(axis=0)[None,:], centers,
                metric='sqeuclidean')
                / centers.shape[0]
            )
        )
        / (
            cdist(centers, centers, metric='sqeuclidean').sum(axis=1)
            / (centers.shape[0] - 1)
        )
    ).sum()


if __name__ == '__main__':
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
        index[k-minK] = ren_liu_wang_yi(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmin() + minK
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
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = ren_liu_wang_yi(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmin() + minK
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
