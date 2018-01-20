import numpy as np
from scipy.spatial.distance import cdist

def fukuyama_sugeno89(data, all_centers, all_labels, m=2) :
    maxK = len(all_centers)
    FS = np.zeros((maxK))

    for k in range(maxK) :
        centers = all_centers[k]
        labels = all_centers[k]
        dist = cdist(data, np.atleast_2d(centers)).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        mu = dist ** (-2.0)
        mu /= np.ones((k+1,1)).dot(np.atleast_2d(mu.sum(axis=0)))
        mu = (mu ** m).T
        Jm = 0
        Km = 0
        for iter1 in range(k+1) :
            Jm += np.sum(mu[:,iter1]*np.sum((data-centers[iter1,:])**2,\
                axis=1))
            Km += np.sum(mu[:,iter1]\
                * np.sum((centers[iter1,:]-np.mean(centers,axis=0))**2))
        FS[k] = Jm - Km

    return np.argmin(FS)+1, FS

if __name__ == '__main__' :
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
    k, FS = fukuyama_sugeno89(data, all_centers, all_labels)
    print(k)
    print(FS)
