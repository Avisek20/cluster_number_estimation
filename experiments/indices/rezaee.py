# Paper Source: B. Rezaee, 2010.
# A cluster validity index for fuzzy clustering.
# Fuzzy Sets and Systems, 161(23), pp. 3014-3025.

# Min number of clusters = 2
# The time complexity is O(nk^2), where n is the number of data points
# and k is the number of clusters.
# To estimate the number of clusters: Use min of vSC


import numpy as np
from scipy.spatial.distance import cdist


def rezaee(data, centers):
    dist = cdist(centers, data, metric='sqeuclidean')
    u = 1 / np.fmax(dist, np.finfo(float).eps)
    u = u / u.sum(axis=0)

    comp = ((u ** 2) * dist).sum()

    h = -(u * np.log(u)).sum(axis=0)
    k = centers.shape[0]
    sep = 0
    for iter1 in range(k) :
        for iter2 in range(iter1+1,k) :
            if iter1 == iter2:
                continue
            sep = sep + (
                np.minimum(u[iter1,:], u[iter2,:]) * h
            ).sum()
    sep = (4 * sep.sum()) / (k * (k - 1))

    return (sep, comp)


if __name__ == '__main__':
    import time
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    sep = np.zeros((maxK-minK))
    comp = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        sep[k-minK], comp[k-minK] = rezaee(
            data, km1.cluster_centers_
        )
    print(sep/ sep.max(), comp/ comp.max())
    index = (sep / sep.max()) + (comp / comp.max())
    print(index)
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

    sep = np.zeros((maxK-minK))
    comp = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        sep[k-minK], comp[k-minK] = rezaee(
            data, km1.cluster_centers_
        )
    print(sep/ sep.max(), comp/ comp.max())
    index = (sep / sep.max()) + (comp / comp.max())
    print(index)
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

    sep = np.zeros((maxK-minK))
    comp = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        sep[k-minK], comp[k-minK] = rezaee(
            data, km1.cluster_centers_
        )
    print(sep/ sep.max(), comp/ comp.max())
    index = (sep / sep.max()) + (comp / comp.max())
    print(index)
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

    sep = np.zeros((maxK-minK))
    comp = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        sep[k-minK], comp[k-minK] = rezaee(
            data, km1.cluster_centers_
        )
    print(sep/ sep.max(), comp/ comp.max())
    index = (sep / sep.max()) + (comp / comp.max())
    print(index)
    est_k = index.argmin() + minK
    print('For 3 Gaussians:\nSelected k =', est_k)
    axisArray[0,3].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,3].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,3].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,3].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,3].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
