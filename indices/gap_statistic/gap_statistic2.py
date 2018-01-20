import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def generate_reference_data(data, B, method='pca') :
    if method == 'uniform' :
        reference_data = np.random.uniform(low=np.amin(data,0), \
            high=np.amax(data,0), size=(B, data.shape[0], data.shape[1]))
    elif method == 'pca' :
        pca1 = PCA(n_components=data.shape[1])
        proj_data = pca1.fit_transform(data)
        reference_data_proj = np.random.uniform(low=np.amin(proj_data,0),\
            high=np.amax(proj_data,0), size=(B, proj_data.shape[0], \
                proj_data.shape[1]))
        reference_data = pca1.inverse_transform(reference_data_proj)
    else :
        print('ERROR : Incorrect argument "method"')
        return
    return reference_data

def intra_cluster_distance(data, centers, labels) :
    dist1 = 0
    k = centers.shape[0]
    for iter1 in range(k) :
        #dist1 += np.sum(pairwise_distances[labels==iter1,:][:,labels==iter1]) \
        #    / np.sum(labels==iter1)
        dist1 += np.sum( (data[labels==iter1,:] - centers[iter1,:])**2 )
    return dist1

def cluster_reference(reference_data, k_range, B) :
    #J0 = np.zeros((B,k_range.shape[0]))
    Wk_ref = np.zeros((B,k_range.shape[0]))
    for b in range(B) :
        #tmp_pairwise_distances = cdist(reference_data[b,:,:], \
        #    reference_data[b,:,:])
        for k in k_range :
            km1 = KMeans(n_clusters=k, n_init=3, max_iter=300, tol=1e-16)\
                .fit(reference_data[b,:,:])
            #Wk_ref[b,k-1] = -km1.score(reference_data[b,:,:])
            Wk_ref[b,k-1] = intra_cluster_distance(reference_data[b,:,:], \
                km1.cluster_centers_, km1.labels_)
    return Wk_ref#, J0

def gap_statistic(data, all_centers, all_labels, B=10, method='pca') :

    #pairwise_distances = cdist(data, data)

    max_k = len(all_labels)
    k_range = np.arange(1, max_k+1)

    Wk = np.zeros((max_k))
    labels = np.zeros(data.shape[0])
    for iter1 in range(max_k) :
        labels = all_labels[iter1]
        centers = all_centers[iter1]
        Wk[iter1] = intra_cluster_distance(data, centers, labels)

    #B = 10
    reference_data = generate_reference_data(data, B, method=method)
    #Wk_ref, ref_J0 = cluster_reference(reference_data, k_range, B)
    Wk_ref = cluster_reference(reference_data, k_range, B)

    # Compute the estimated gap statistic
    logWk_ref = np.log(Wk_ref)
    l = np.mean(logWk_ref, 0)
    logWk = np.log(Wk)
    Gap = l - logWk
    sdk = (1.0/np.sqrt(B))*np.linalg.norm(logWk_ref - l, axis=0)
    sk = np.sqrt(1+(1/B))*sdk

    estimated_k = np.where((Gap[1:] - Gap[0:-1]) <= sk[1:])[0][0] + 1

    #print('log Gap k :', estimated_k)

    Gap_nolog = np.mean(Wk_ref,0) - Wk
    sdk_nolog = (1.0 / np.sqrt(B)) * np.linalg.norm(Wk_ref - np.mean(Wk_ref,0) \
        , axis=0)
    sk_nolog = np.sqrt(1+(1/B))*sdk_nolog
    estimated_k_nolog = np.where((Gap_nolog[1:] - Gap_nolog[0:-1])\
        <= sk_nolog[1:])[0][0] + 1
    #print('Gap k :',estimated_k_nolog)

    return estimated_k, logWk, logWk_ref, estimated_k_nolog, Wk, Wk_ref


if __name__ == '__main__' :
    import time
    from sklearn.cluster import KMeans
    data = np.vstack(( np.vstack(( \
        np.random.normal(loc = [0,0], scale=1, size=(200,2)), \
        np.random.normal(loc = [10,0], scale=1, size=(200,2)) \
        )),
        np.random.normal(loc = [5,10], scale=1, size=(200,2))
        ))

    all_centers = []
    all_labels = []
    maxK = int((data.shape[0])**0.5)
    for iter1 in range(1,maxK+1) :
        km1 = KMeans(n_clusters=iter1, max_iter=300, n_init=10, tol=1e-16).fit(data)
        all_centers.append(km1.cluster_centers_)
        all_labels.append(km1.labels_)
    start = time.time()
    k, logWk, logWk_ref, estimated_k_nolog, Wk, Wk_ref = \
        gap_statistic(data, all_centers, all_labels)
    print(k)
    print(time.time() - start)
