'''Last Major Leap

Paper Source: Gupta A., Datta S., Das S., "Fast automatic estimation of the
number of clusters from the minimum inter-center distance for k-means
clustering", Pattern Recognition Letters, vol. 116, pp. 72-79, 2018.
'''

# Author: Avisek Gupta


import numpy as np
from scipy.spatial.distance import cdist


def last_major_leap(all_centers):
    '''
    The Last Major Leap:
        Method to identify the number of clusters
        between 1 and sqrt(n_data_points)

    Parameters
    ----------

    all_centers : list or tuple, shape (n_centers)
        All cluster centers from k=2 to k=ceil(sqrt(n_data_points))

    Returns
    -------

    k_est: int
        The estimated number of clusters

    min_dist: array, shape(k_max - 1)
        The index values at each k from k=2 to k=ceil(sqrt(n_data_points))

    '''

    k_min = 2
    k_max = len(all_centers) + 1

    min_dist = np.zeros((k_max - 1))
    for i in range(k_min, k_max + 1):
        dist = cdist(
            all_centers[i - k_min], all_centers[i - k_min],
            metric='sqeuclidean'
        )
        np.fill_diagonal(dist, +np.inf)
        min_dist[i - k_min] = dist.min()

    k_est = 1
    for i in range(min_dist.shape[0] - k_min, 0 - 1, -1):
        if (min_dist[i] * 0.5) > (min_dist[i+1:]).max():
            k_est = i + k_min
            break

    return k_est, min_dist


if __name__ == '__main__':
    # DEBUG

    '''
    k_min = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    k_max = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    all_centers = []
    for k in range(k_min, k_max):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        all_centers.append(km1.cluster_centers_)
    est_k, index = last_major_leap(all_centers)
    print('For Iris:\nSelected k =', est_k)

    import matplotlib.pyplot as plt
    fig, axisArray = plt.subplots(2, 2)
    axisArray[0, 0].scatter(data[:, 2], data[:, 3], marker='x', c='gray')
    axisArray[1, 0].plot(
        np.arange(index.shape[0]) + k_min, index, linewidth=0.3, c='k'
    )
    axisArray[1, 0].scatter(
        np.arange(index.shape[0]) + k_min, index, marker='x', c='b'
    )
    axisArray[1, 0].scatter(est_k, index[est_k - k_min], marker='x', c='r')
    axisArray[1, 0].set_xticks(np.arange(index.shape[0]) + k_min)

    data = np.vstack((
        np.random.normal(loc=np.array([0, 0]), scale=1, size=(50, 2)),
        np.random.normal(loc=np.array([20, 20]), scale=1, size=(50, 2)),
        np.random.normal(loc=np.array([0, 20]), scale=1, size=(50, 2)),
        np.random.normal(loc=np.array([20, 0]), scale=1, size=(50, 2)),
        np.random.normal(loc=np.array([10, 10]), scale=1, size=(50, 2))
    ))
    k_max = int(np.ceil(data.shape[0] ** 0.5))

    all_centers = []
    for k in range(k_min, k_max):
        km1 = KMeans(n_clusters=k, n_init=30, max_iter=300, tol=1e-9).fit(data)
        all_centers.append(km1.cluster_centers_)
    est_k, index = last_major_leap(all_centers)
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0, 1].scatter(data[:, 0], data[:, 1], marker='x', c='gray')
    axisArray[1, 1].plot(
        np.arange(index.shape[0]) + k_min, index, linewidth=0.3, c='k'
    )
    axisArray[1, 1].scatter(
        np.arange(index.shape[0]) + k_min, index, marker='x', c='b'
    )
    axisArray[1, 1].scatter(est_k, index[est_k - k_min], marker='x', c='r')
    axisArray[1, 1].set_xticks(np.arange(index.shape[0]) + k_min)

    plt.show()
    '''
