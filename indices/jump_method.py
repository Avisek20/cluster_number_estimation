# Paper Source: Sugar, C.A. and James, G.M., 2003.
# Finding the number of clusters in a dataset: An information-theoretic approach.
# Journal of the American Statistical Association, 98(463), pp.750-763.

# Min number of clusters = 1
# The time complexity is O(nk), where n is the number of data points and k is the number of clusters.
# To estimate the number of clusters: Use maximum of the first difference


import numpy as np
from scipy.spatial.distance import cdist


def jump_method(d0, d1, y) :
	return d1 ** (-y) - d0 ** (-y)


if __name__ == '__main__':
    minK = 2
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    score = []
    for k in range(minK-1,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            score.append(-km1.score(data) / data.shape[1])
        elif k == 2:
            score.append(-km1.score(data) / data.shape[1])
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
        else:
            score[0] = score[1]
            score[1] = -km1.score(data) / data.shape[1]
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
    from elbow_method import elbow_method
    #est_k = elbow_method(index) + minK
    est_k = index.argmax() + minK
    print('Selected k =', est_k)

    import matplotlib.pyplot as plt
    fig, axisArray = plt.subplots(2,4)
    axisArray[0,0].scatter(data[:,2], data[:,3], marker='x', c='gray')
    axisArray[1,0].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,0].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,0].scatter(est_k, index[est_k-minK], marker='x', c='r')

    data = np.vstack(( np.vstack(( np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([10,10]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,20]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    score = []
    for k in range(minK-1,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            score.append(-km1.score(data) / data.shape[1])
        elif k == 2:
            score.append(-km1.score(data) / data.shape[1])
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
        else:
            score[0] = score[1]
            score[1] = -km1.score(data) / data.shape[1]
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
    est_k = index.argmax() + minK
    print('Selected k =', est_k)
    axisArray[0,1].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,1].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,1].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,1].scatter(est_k, index[est_k-minK], marker='x', c='r')


    data = np.vstack(( np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,20]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    score = []
    for k in range(minK-1,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            score.append(-km1.score(data) / data.shape[1])
        elif k == 2:
            score.append(-km1.score(data) / data.shape[1])
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
        else:
            score[0] = score[1]
            score[1] = -km1.score(data) / data.shape[1]
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
    est_k = index.argmax() + minK
    print('Selected k =', est_k)
    axisArray[0,2].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,2].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,2].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,2].scatter(est_k, index[est_k-minK], marker='x', c='r')

    data = np.vstack(( np.vstack((
        np.random.normal(loc=np.array([0,0]), scale=1, size=(50,2)),
        np.random.normal(loc=np.array([0,20]), scale=1, size=(50,2)) )),
        np.random.normal(loc=np.array([20,0]), scale=1, size=(50,2))
    ))
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    score = []
    for k in range(minK-1,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        if k < 2:
            score.append(-km1.score(data) / data.shape[1])
        elif k == 2:
            score.append(-km1.score(data) / data.shape[1])
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
        else:
            score[0] = score[1]
            score[1] = -km1.score(data) / data.shape[1]
            index[k-minK] = jump_method(
                score[0], score[1], 1
            )
    est_k = index.argmax() + minK
    print('Selected k =', est_k)
    axisArray[0,3].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,3].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,3].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,3].scatter(est_k, index[est_k-minK], marker='x', c='r')

    plt.show()
