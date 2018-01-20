import numpy as np
from sklearn.metrics import calinski_harabaz_score

def calinski_harabasz(data, all_centers, all_labels) :
    N = data.shape[0]

    CH = np.zeros((len(all_labels)-1))
    for iter2 in range(len(all_labels)-1) :
        labels = all_labels[iter2+1]
        CH[iter2] = calinski_harabaz_score(data, labels)
    return np.argmax(CH)+2, CH
