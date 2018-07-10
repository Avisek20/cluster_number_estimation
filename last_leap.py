# Min number of clusters = 2
# The time complexity is O(k^2), where k is the number of clusters.


import numpy as np
from scipy.spatial.distance import cdist


def last_leap(all_centers):
    minK = 2
    maxK = len(all_centers) + 1

    min_dist = np.zeros((maxK - 1))
    for iter1 in range(minK, maxK+1):
        dist = cdist(
            all_centers[iter1-minK], all_centers[iter1-minK],
            metric='sqeuclidean'
        )
        np.fill_diagonal(dist, +np.inf)
        min_dist[iter1-minK] = dist.min()

    k_est1 = (
        (min_dist[0:-1] - min_dist[1:]) / min_dist[0:-1]
    ).argmax() + minK

    rest_of_the_data = min_dist[k_est1-minK+1:]
    if ((min_dist[k_est1-2] * 0.5) < rest_of_the_data).sum() > 0 :
        k_est1 = 1

    return k_est1, min_dist


if __name__ == '__main__':
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    all_centers = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        all_centers.append(km1.cluster_centers_)
    est_k, index = last_leap(all_centers)
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

    all_centers = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        all_centers.append(km1.cluster_centers_)
    est_k, index = last_leap(all_centers)
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
