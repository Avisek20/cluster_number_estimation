# Paper Source: Fukuyama, Y., 1989.
# A new method of choosing the number of clusters
# for the fuzzy c-mean method. In Proc. 5th Fuzzy Syst. Symp.,
# 1989 (pp. 247-250).

# Min number of clusters = 1
# The time complexity is O(nk), where n is the number of data points and k is the number of clusters.
# To estimate the number of clusters: Use minimum FS


import numpy as np
from scipy.spatial.distance import cdist


def fukuyama_sugeno(data, centers, m=2):
    if centers.ndim == 1:
        centers = centers[None,:]
    dist = np.fmax(cdist(centers, data, metric='sqeuclidean'), np.finfo(np.float).eps)
    u = (1 / dist)
    um = (u / u.sum(axis=0)) ** m
    return ((um * dist).sum() - cdist(
            centers, centers.mean(axis=0)[None,:], metric='sqeuclidean'
    ).sum())


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    minK = 1
    index = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = fukuyama_sugeno(
            data, km1.cluster_centers_
        )
    from elbow_method import elbow_method
    est_k = elbow_method(index) + minK
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
        index[k-minK] = fukuyama_sugeno(
            data, km1.cluster_centers_
        )
    est_k = elbow_method(index) + minK
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
