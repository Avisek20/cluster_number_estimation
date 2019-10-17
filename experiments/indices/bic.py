# Important Note: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
# Paper Source: (1) Zhao, Q., Xu, M. and Fränti, P., 2008, November.
# Knee point detection on Bayesian information criterion.
# In Tools with Artificial Intelligence, 2008. ICTAI'08.
# 20th IEEE International Conference on (Vol. 2, pp. 431-438). IEEE.
# (2) G. Schwarz, 1978. Estimating the dimension of a model.
# Ann. Stat. 6 (2) (1978) 461–464.
# (3) Mehrjou A., Hosseini R. and Araabi B. N., 2016.
# Improved Bayesian information criterion for mixture model selection.
# Pattern Recognition Letters, 69, pp. 22-27.

# Min number of clusters = 2
# The time complexity is O(nk), where n is the number of data points,
# and k is the number of clusters
# To estimate the number of clusters: Use max of BIC


import numpy as np
from scipy.spatial.distance import cdist


def bic(data, centers, labels):
    ni = np.fmax(
        np.unique(labels, return_counts=True)[1], np.finfo(float).eps
    )
    labelmask = np.zeros((centers.shape[0], data.shape[0]))
    for i in range(centers.shape[0]):
        labelmask[i, labels==i] = 1
    '''
    denom = ni - centers.shape[0]
    denom[denom==0] = np.finfo(float).eps
    sigmai = np.fmax(
        (cdist(centers, data, metric='sqeuclidean')
        * labelmask).sum(axis=1)
        / denom
        , np.finfo(float).eps
    )
    '''
    denom = data.shape[1] / (data.shape[0] - centers.shape[0])
    sigma = (cdist(centers, data, metric='sqeuclidean') * labelmask).sum() / denom
    return (
        (ni * np.log(ni)) - (ni * data.shape[0])
        - (0.5 * ni * data.shape[1]) * np.log(2 * np.pi * sigma)
        - ((ni - 1) * data.shape[1] * 0.5)
    ).sum() - (0.5 * centers.shape[0] * np.log(data.shape[0]) * (data.shape[1] + 1))
    '''
    return ((
        ((ni * np.log(ni)) - (ni * data.shape[0]))
        - (0.5 * ni * data.shape[1] * np.log(2*np.pi))
        - (0.5 * ni * np.log(sigmai)) - (0.5 * (ni - centers.shape[0]))
    ).sum() - (0.5 * centers.shape[0] * np.log(data.shape[0])))

    return ((-2 * (
        (ni * np.log(ni)) - (ni * np.log(data.shape[0]))
        - (ni * data.shape[1] * 0.5 * np.log(2*np.pi))
        - (ni * np.log(sigmai) * 0.5) - ((ni - centers.shape[0]) * 0.5)
    ).sum()) + (centers.shape[0] * np.log(data.shape[0])))
    '''


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
        index[k-minK] = bic(
            data, km1.cluster_centers_, km1.labels_
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
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK] = bic(
            data, km1.cluster_centers_, km1.labels_
        )
    est_k = index.argmax() + minK
    print('For 5 Gaussians:\nSelected k =', est_k)

    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
