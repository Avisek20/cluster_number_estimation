# Paper Source: Hartigan, J.A., 1985.
# Statistical theory in clustering.
# Journal of classification, 2(1), pp.63-76.

# Min number of clusters = 2
# The time complexity is O(nk), where n is the number of data points and k is the number of clusters.
# To estimate the number of clusters: Use minimum HART_85


import numpy as np
from scipy.spatial.distance import cdist


def hartigan_85(data, centers1, centers2):
    if centers1.ndim == 1:
        return (data.shape[0] - centers1.shape[0] - 1) * ((
            cdist(centers1[None,:], data, metric='sqeuclidean').sum()
            / cdist(centers2, data, metric='sqeuclidean').min(axis=0).sum()
        ) - 1)
    else:
        return (data.shape[0] - centers1.shape[0] - 1) * ((
            cdist(centers1, data, metric='sqeuclidean').min(axis=0).sum()
            / cdist(centers2, data, metric='sqeuclidean').min(axis=0).sum()
        ) - 1)


if __name__ == '__main__':
    minK = 1
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    centers_arr = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            centers_arr.append(np.array(km1.cluster_centers_))
        elif k == 2:
            centers_arr.append(np.array(km1.cluster_centers_))
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
        else:
            centers_arr[0] = np.array(centers_arr[1])
            centers_arr[1] = np.array(km1.cluster_centers_)
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
    from elbow_method import elbow_method
    est_k = elbow_method(index) + minK
    print('For Iris:\nSelected k =', est_k)

    import matplotlib.pyplot as plt
    fig, axisArray = plt.subplots(2,5)
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
    centers_arr = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            centers_arr.append(np.array(km1.cluster_centers_))
        elif k == 2:
            centers_arr.append(np.array(km1.cluster_centers_))
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
        else:
            centers_arr[0] = np.array(centers_arr[1])
            centers_arr[1] = np.array(km1.cluster_centers_)
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
    est_k = elbow_method(index) + minK
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
    centers_arr = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            centers_arr.append(np.array(km1.cluster_centers_))
        elif k == 2:
            centers_arr.append(np.array(km1.cluster_centers_))
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
        else:
            centers_arr[0] = np.array(centers_arr[1])
            centers_arr[1] = np.array(km1.cluster_centers_)
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
    est_k = elbow_method(index) + minK
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
    centers_arr = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            centers_arr.append(np.array(km1.cluster_centers_))
        elif k == 2:
            centers_arr.append(np.array(km1.cluster_centers_))
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
        else:
            centers_arr[0] = np.array(centers_arr[1])
            centers_arr[1] = np.array(km1.cluster_centers_)
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
    est_k = elbow_method(index) + minK
    print('For 3 Gaussians:\nSelected k =', est_k)
    axisArray[0,3].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,3].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,3].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,3].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,3].set_xticks(np.arange(index.shape[0])+minK)


    data = np.vstack(( np.vstack(( np.vstack(( np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([20,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([40,0]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([40,20]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    index = np.zeros((maxK-minK))
    centers_arr = []
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            centers_arr.append(np.array(km1.cluster_centers_))
        elif k == 2:
            centers_arr.append(np.array(km1.cluster_centers_))
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
        else:
            centers_arr[0] = np.array(centers_arr[1])
            centers_arr[1] = np.array(km1.cluster_centers_)
            index[k-minK-1] = hartigan_85(
                data, centers_arr[0], centers_arr[1]
            )
    est_k = elbow_method(index) + minK
    print('For 5 Gaussians:\nSelected k =', est_k)
    axisArray[0,4].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,4].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,4].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,4].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,4].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
