# Paper Source: R. Tibshirani, G. Walther, T. Hastie,
# Estimating the number of clusters in a data set via the gap statistic,
# Journal of the Royal Statistical Society: Series B (Statistical Methodology)
# 63 (2) (2001) 411–423.

# Min number of clusters = 1
# The time complexity is O(nk), where n is the number of data points
# and k is the number of clusters.
# To estimate the number of clusters: Use max of GAP


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def generate_reference_data(data, B, method='pca'):
    if method == 'uniform':
        reference_data = np.random.uniform(
            low=data.min(axis=0), high=data.max(axis=0),
            size=(B, data.shape[0], data.shape[1])
        )
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca1 = PCA(n_components=data.shape[1])
        proj_data = pca1.fit_transform(data)
        reference_data_proj = np.random.uniform(
            low=proj_data.min(axis=0), high=proj_data.max(axis=0),
            size=(B, proj_data.shape[0], proj_data.shape[1])
        )
        reference_data = pca1.inverse_transform(reference_data_proj)
    else :
        print('ERROR : Incorrect argument "method"')
        return
    return reference_data


def gap_statistic(data, centers, permuted_data, B=30):
    if centers.ndim == 1:
        centers = centers[None,:]
    k = centers.shape[0]
    wk = cdist(centers, data, metric='sqeuclidean').min(axis=0).sum()
    wk_permuted = np.zeros((B))
    for b in range(B):
        #print(k, permuted_data[b,:,:].shape)
        km1 = KMeans(
            n_clusters=k, n_init=2, max_iter=80, tol=1e-6
        ).fit(permuted_data[b,:,:])
        wk_permuted[b] = -km1.score(permuted_data[b,:,:])
    log_wk_permuted = np.log(wk_permuted)
    return (log_wk_permuted.mean() - np.log(wk)), (((
            ((log_wk_permuted - log_wk_permuted.mean()) ** 2).mean()
    ) ** 0.5) * ((1 + (1 / B)) ** 0.5))

if __name__ == '__main__':
    minK = 1
    from sklearn.datasets import load_iris
    data = load_iris().data
    maxK = int(np.ceil(data.shape[0] ** 0.5))

    B = 10
    permuted_data = generate_reference_data(data, B=B, method='pca')
    from sklearn.cluster import KMeans
    index = np.zeros((maxK-minK))
    sk = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK], sk[k-minK] = gap_statistic(
            data, km1.cluster_centers_, permuted_data, B=B
        )
    est_k = np.where(index[0:-1] >= (index[1:] - sk[1:]))[0][0] + minK
    #from elbow_method import elbow_method
    #est_k = elbow_method(index) + minK
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


    permuted_data = generate_reference_data(data, B=B, method='pca')
    index = np.zeros((maxK-minK))
    sk = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK], sk[k-minK] = gap_statistic(
            data, km1.cluster_centers_, permuted_data, B=B
        )
    est_k = np.where(index[0:-1] >= (index[1:] - sk[1:]))[0][0] + minK
    #est_k = elbow_method(index) + minK
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


    permuted_data = generate_reference_data(data, B=B, method='pca')
    index = np.zeros((maxK-minK))
    sk = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK], sk[k-minK] = gap_statistic(
            data, km1.cluster_centers_, permuted_data, B=B
        )
    est_k = np.where(index[0:-1] >= (index[1:] - sk[1:]))[0][0] + minK
    #est_k = elbow_method(index) + minK
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


    permuted_data = generate_reference_data(data, B=B, method='pca')
    index = np.zeros((maxK-minK))
    sk = np.zeros((maxK-minK))
    for k in range(minK,maxK):
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-9
        ).fit(data)
        index[k-minK], sk[k-minK] = gap_statistic(
            data, km1.cluster_centers_, permuted_data, B=B
        )
    est_k = np.where(index[0:-1] >= (index[1:] - sk[1:]))[0][0] + minK
    #est_k = elbow_method(index) + minK
    print('For 3 Gaussians:\nSelected k =', est_k)
    axisArray[0,3].scatter(data[:,0], data[:,1], marker='x', c='gray')
    axisArray[1,3].plot(np.arange(index.shape[0])+minK, index, linewidth=0.3, c='k')
    axisArray[1,3].scatter(np.arange(index.shape[0])+minK, index, marker='x', c='b')
    axisArray[1,3].scatter(est_k, index[est_k-minK], marker='x', c='r')
    axisArray[1,3].set_xticks(np.arange(index.shape[0])+minK)

    plt.show()
