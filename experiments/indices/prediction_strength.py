# Paper Source: R. Tibshirani, G. Walther, 2005.
# Cluster Validation by Prediction Strength,
# Journal of Computational and Graphical Statistics, 14(3) pp. 511–528.

# Min number of clusters = 1
# The time complexity is O(n_fold.n^2.k),
# where n is the number of data points,
# n_fold is the number of folds in cross-validation,
# and k is the number of clusters.
# To estimate the number of clusters: Use Prediction Threshold


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def get_crossvalidation_data(data, n_fold=2):
    permuted_data = data[np.random.permutation(data.shape[0]),:]
    xdatas = []
    for iter1 in range(n_fold):
        if iter1 == 0:
            xdatas.append((
                permuted_data[data.shape[0]//n_fold:,:],
                permuted_data[0:data.shape[0]//n_fold,:]
            ))
        elif iter1 == (n_fold-1):
            xdatas.append((
                permuted_data[0:iter1*data.shape[0]//n_fold,:],
                permuted_data[iter1*data.shape[0]//n_fold:,:]
            ))
        else:
            xdatas.append((
                np.vstack((
                    permuted_data[0:iter1*data.shape[0]//n_fold, :],
                    permuted_data[(iter1+1)*data.shape[0]//n_fold:, :]
                )),
                permuted_data[
                    iter1*data.shape[0]//n_fold
                    :(iter1+1)*data.shape[0]//n_fold, :
                ]
            ))
    return xdatas


def prediction_strength(xdatas, n_clusters):
    PS = 0
    for train, test in xdatas:
        km_train = KMeans(
            n_clusters=n_clusters, max_iter=80, n_init=3, tol=1e-6
        ).fit(train)
        km_test = KMeans(
            n_clusters=n_clusters, max_iter=80, n_init=3, tol=1e-6
        ).fit(test)
        train_labels = cdist(
            km_train.cluster_centers_, test
        ).argmin(axis=0)

        ps_k = +np.inf
        for iterk in range(n_clusters):
            co_occurence = np.outer(
                km_test.labels_==iterk, train_labels==iterk
            )
            np.fill_diagonal(co_occurence, 0)
            ps_k = min(
                ps_k, co_occurence.sum() / (km_test.labels_==iterk).sum()
            )
        PS += ps_k
    return PS / len(xdatas)

if __name__ == '__main__':
    '''
    data = np.arange(40).reshape(20,2)
    k = 3
    data = np.hstack((
        data, np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2]]).T
    ))
    xdatas = get_crossvalidation_data(data, n_fold=2)
    prediction_strength(xdatas, k=k)
    '''
    if __name__ == '__main__':
        import time
        minK = 1
        from sklearn.datasets import load_iris
        data = load_iris().data
        maxK = int(np.ceil(data.shape[0] ** 0.5))

        xdatas = get_crossvalidation_data(data, n_fold=2)
        index = np.zeros((maxK-minK))
        for k in range(minK,maxK):
            index[k-minK] = prediction_strength(
                xdatas, n_clusters=k
            )
        #print('index', index, index.mean(), 0.8*index.mean())
        est_k = np.where(index > index.mean())[0][-1] + minK
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

        xdatas = get_crossvalidation_data(data, n_fold=2)
        index = np.zeros((maxK-minK))
        for k in range(minK,maxK):
            index[k-minK] = prediction_strength(
                xdatas, n_clusters=k
            )
        #print('index', index, index.mean(), 0.8*index.mean())
        est_k = np.where(index > index.mean())[0][-1] + minK
        print('For 5 Gaussians:\nSelected k =', est_k)
        axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
        axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
        axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
        axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')
        axisArray[1,1].set_xticks(np.arange(index.shape[0])+minK)

        plt.show()
