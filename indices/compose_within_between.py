# Paper Source: Rezaee, M.R., Lelieveldt, B.P. and Reiber, J.H., 1998.
# A new cluster validity index for the fuzzy c-mean.
# Pattern recognition letters, 19(3-4), pp.237-246.

# Min number of clusters = 2
# The time complexity is O(nk), where n is the number of data points and k is the number of clusters.
# To estimate the number of clusters: Use minimum of CWB


import numpy as np
from scipy.spatial.distance import cdist

def compose_within_between(data, centers, centerskmax):
    k = centers.shape[0]
    n = data.shape[0]

    dist = np.fmax(cdist(centers, data), np.finfo(float).eps)
    u = 1 / dist
    u = u / u.sum(axis=0)
    sigma = np.zeros((k, data.shape[1]))
    for iter1 in range(k):
        sigma[iter1,:] = (
            ((data - centers[iter1,:]) ** 2).T * u[iter1,:]
        ).mean(axis=1)
    sigma_x = ((data - data.mean(axis=0)) ** 2).mean(axis=0)
    Scat = np.linalg.norm(sigma, axis=1).mean() / np.linalg.norm(sigma_x)

    dist_centers = cdist(centers, centers)
    dmax = dist_centers.max()
    np.fill_diagonal(dist_centers, np.inf)
    dmin = dist_centers.min()
    np.fill_diagonal(dist_centers, 0)
    Dis = (dmax / dmin) * (1 / dist_centers.sum(axis=1)).sum()

    dist_centerskmax = cdist(centerskmax, centerskmax)
    dmaxkmax = dist_centerskmax.max()
    np.fill_diagonal(dist_centerskmax, np.inf)
    dminkmax = dist_centerskmax.min()
    np.fill_diagonal(dist_centerskmax, 0)
    alpha = (dmaxkmax / dminkmax) * (1 / dist_centerskmax.sum(axis=1)).sum()

    return alpha*Scat + Dis


if __name__ == '__main__':
    import time
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    km1 = KMeans(n_clusters=maxK, n_init=30, max_iter=300, tol=1e-9).fit(data)
    centerskmax = np.array(km1.cluster_centers_)
    for k in range(minK,maxK-1):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = compose_within_between(
            data, km1.cluster_centers_, centerskmax
        )
    start = time.time()
    index[maxK-1-minK] = compose_within_between(
        data, centerskmax, centerskmax
    )
    print(time.time() - start)
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
    km1 = KMeans(n_clusters=maxK, n_init=30, max_iter=300, tol=1e-9).fit(data)
    centerskmax = np.array(km1.cluster_centers_)
    for k in range(minK,maxK-1):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = compose_within_between(
            data, km1.cluster_centers_, centerskmax
        )
    index[maxK-1-minK] = compose_within_between(
        data, centerskmax, centerskmax
    )
    est_k = index.argmin() + minK
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
