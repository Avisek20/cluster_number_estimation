import os
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from indices.aic import aic
from indices.bic import bic
from indices.calinski_harabasz import calinski_harabasz
from indices.classification_entropy import classification_entropy
from indices.compose_within_between import compose_within_between
from indices.davies_bouldin import davies_bouldin
from indices.dunn import dunn
from indices.elbow_method import elbow_method
from indices.fukuyama_sugeno import fukuyama_sugeno
from indices.fuzzy_hypervolume import fuzzy_hypervolume
from indices.gap_statistic import gap_statistic
from indices.gap_statistic import generate_reference_data
from indices.halkidi_vazirgannis import halkidi_vazirgannis
from indices.hartigan_85 import hartigan_85
from indices.I_index import I_index
from indices.jump_method import jump_method
from indices.last_leap import last_leap
from indices.last_major_leap import last_major_leap
from indices.modified_partition_coefficient import modified_partition_coefficient
from indices.partition_coefficient import partition_coefficient
from indices.partition_index import partition_index
from indices.pbmf import pbmf
from indices.pcaes import pcaes
from indices.prediction_strength import prediction_strength
from indices.prediction_strength import get_crossvalidation_data
from indices.ren_liu_wang_yi import ren_liu_wang_yi
from indices.rezaee import rezaee
from indices.silhouette import silhouette
from indices.slope_statistic import slope_statistic
from indices.xie_beni import xie_beni
from indices.xu_index import xu_index
from indices.zhao_xu_franti import zhao_xu_franti

indices = [
    aic, bic, calinski_harabasz, classification_entropy,
    compose_within_between, davies_bouldin, dunn, elbow_method,
    fukuyama_sugeno, fuzzy_hypervolume, gap_statistic,
    halkidi_vazirgannis, hartigan_85, I_index, jump_method,
    last_leap, last_major_leap,
    modified_partition_coefficient, partition_coefficient,
    partition_index, pbmf, pcaes, prediction_strength,
    ren_liu_wang_yi, rezaee, silhouette, slope_statistic,
    xie_beni, xu_index, zhao_xu_franti
]

# Output: for every index, the cluster number from different methods.
# Plot all outputs at the estimated k

output_dir = 'output/'

data_path = './'
data_dirs = ['data_well_separated', 'data_diff_spread',
    'data_slightly_diff_points', 'data_very_diff_points',
    'data_slight_overlap2', 'data_high_overlap2']
cluster_dirs = ['kmeans_cluster_data_well_separated',
'kmeans_cluster_data_diff_spread', 'kmeans_cluster_data_slightly_diff_points',
'kmeans_cluster_data_very_diff_points', 'kmeans_cluster_data_slight_overlap2',
'kmeans_cluster_data_high_overlap2']
n_data_dirs = [30, 30, 30, 30, 30, 30]
n_data_sets = [25, 25, 25, 25, 5, 5]

for idx_data_dir in range(2,len(data_dirs)):
    results_dir = 'results_km_'+data_dirs[idx_data_dir]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for idx_n_data_dir in range(n_data_dirs[idx_data_dir]):
        results_dir2 = data_dirs[idx_data_dir]+'_'+str(idx_n_data_dir)
        if not os.path.exists(results_dir+'/'+results_dir2):
            os.makedirs(results_dir+'/'+results_dir2)

        for idx_n_data_set in range(n_data_sets[idx_data_dir]):
            data = np.loadtxt(
                data_path+data_dirs[idx_data_dir]
                +'/'+data_dirs[idx_data_dir]+'_'+str(idx_n_data_dir)
                +'/'+'data_'+str(idx_n_data_set)+'.txt'
            )
            print(data_dirs[idx_data_dir], idx_n_data_dir, idx_n_data_set, data.shape)
            labels = data[:,-1]
            k_true = len(np.unique(labels))
            data = data[:,0:-1]
            k_max = int(np.ceil(data.shape[0] ** 0.5))

            centers = []
            labels = []
            for k in range(1, k_max+1):
                centers.append(np.loadtxt(cluster_dirs[idx_data_dir]+'/'+data_dirs[idx_data_dir]+'_'+str(idx_n_data_dir)+'/'+'data_'+str(idx_n_data_set)+'_k_'+str(k)+'_centers.txt'))
                labels.append(np.loadtxt(cluster_dirs[idx_data_dir]+'/'+data_dirs[idx_data_dir]+'_'+str(idx_n_data_dir)+'/'+'data_'+str(idx_n_data_set)+'_k_'+str(k)+'_labels.txt'))

            pairwise_distances = cdist(data, data)

            est_k = np.zeros((len(indices)))
            for iter_idx in range(len(indices)):
                #print('index',iter_idx,'...')
                if iter_idx in [26]: # Slope calculated while doing SIL
                    continue
                if iter_idx in [1,2]: # BIC, CH
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1], labels[k-1])
                    est_k[iter_idx] = idx_val.argmax() + k_min
                elif iter_idx in [11, 23]: # HVZ, RLWY
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1], labels[k-1])
                    est_k[iter_idx] = idx_val.argmin() + k_min
                elif iter_idx in [4]: #CWB
                    centerskmax = centers[-1]
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1], centerskmax)
                    est_k[iter_idx] = idx_val.argmin() + k_min
                elif iter_idx in [0, 5, 29]: #AIC, DB, ZXF
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1], labels[k-1])
                    est_k[iter_idx] = elbow_method(idx_val) + k_min
                elif iter_idx in [6, 25]: #DUNN, SIL & slope
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](pairwise_distances, labels[k-1])
                    est_k[iter_idx] = idx_val.argmax() + k_min
                    if iter_idx == 25:
                        idx_val = slope_statistic(idx_val, data.shape[1])
                        est_k[iter_idx+1] = idx_val.argmax() + k_min
                elif iter_idx in [7]: # elbow
                    k_min = 1
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        if k == 1:
                            idx_val[k-k_min] = cdist(centers[k-1][None,:], data, metric='sqeuclidean').min(axis=0).sum()
                        else:
                            idx_val[k-k_min] = cdist(centers[k-1], data, metric='sqeuclidean').min(axis=0).sum()
                    est_k[iter_idx] = elbow_method(idx_val) + k_min
                elif iter_idx in [8]: # Fukuyama
                    k_min = 1
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1])
                    est_k[iter_idx] = elbow_method(idx_val) + k_min
                elif iter_idx in [9]: # FHV
                    k_min = 1
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1])
                    est_k[iter_idx] = idx_val.argmin() + k_min
                elif iter_idx in [10]: # Gap
                    k_min = 1
                    B = 3
                    permuted_data = generate_reference_data(data, B=B, method='pca')
                    idx_val = np.zeros((k_max-k_min+1))
                    sk = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min], sk[k-k_min] = gap_statistic(
                            data, centers[k-1], permuted_data, B=B
                        )
                    est_k[iter_idx] = 1
                    sel_idxs = np.where(idx_val[0:-1] >= (idx_val[1:] - sk[1:]))[0]
                    if len(sel_idxs) > 0:
                        est_k[iter_idx] = sel_idxs[0] + k_min
                elif iter_idx in [12]: # Hart85
                    k_min = 1
                    idx_val = np.zeros((k_max-k_min))
                    for k in range(k_min, k_max):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1], centers[k])
                    est_k[iter_idx] = elbow_method(idx_val) + k_min
                elif iter_idx in [13, 17, 18, 20, 21]: #I, MPC, PC, PBMF, PCAES
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1])
                    est_k[iter_idx] = idx_val.argmax() + k_min
                elif iter_idx in [14]: # jump
                    k_min = 2
                    scores = np.zeros((k_max-k_min+2))
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min-1, k_max+1):
                        if k == 1:
                            scores[k-(k_min-1)] = cdist(centers[k-1][None,:], data, metric='sqeuclidean').min(axis=0).sum()
                        else:
                            scores[k-(k_min-1)] = cdist(centers[k-1], data, metric='sqeuclidean').min(axis=0).sum()
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = jump_method(
                            scores[k-k_min]/data.shape[1], scores[k-(k_min-1)]/data.shape[1], 1
                        )
                    est_k[iter_idx] = idx_val.argmax() + k_min
                elif iter_idx in [15, 16]: # LL, LML
                    k_min = 2
                    est_k[iter_idx], tmp = indices[iter_idx](centers[1:])
                elif iter_idx in [19]: # PI
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1])
                    est_k[iter_idx] = elbow_method(idx_val) + k_min
                elif iter_idx in [22]: # Prediction Strength
                    k_min = 1
                    xdatas = get_crossvalidation_data(data, n_fold=2)
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min,k_max+1):
                        idx_val[k-k_min] = prediction_strength(xdatas, n_clusters=k)
                    est_k[iter_idx] = 1
                    sel_idxs = np.where(idx_val > idx_val.mean())[0]
                    if len(sel_idxs) > 0:
                        est_k[iter_idx] = sel_idxs[-1] + k_min
                elif iter_idx in [24]: # Rezaee
                    k_min = 2
                    sep = np.zeros((k_max-k_min+1))
                    comp = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        sep[k-k_min], comp[k-k_min] = indices[iter_idx](data, centers[k-1])
                    idx_val = (sep / sep.max()) + (comp / comp.max())
                    est_k[iter_idx] = idx_val.argmin() + k_min
                elif iter_idx in [3, 27, 28]: # CE, XB, Xu
                    k_min = 2
                    idx_val = np.zeros((k_max-k_min+1))
                    for k in range(k_min, k_max+1):
                        idx_val[k-k_min] = indices[iter_idx](data, centers[k-1])
                    est_k[iter_idx] = idx_val.argmin() + k_min

            np.savetxt(results_dir+'/'+results_dir2+'/results_'+str(idx_n_data_set), est_k, fmt='%d')
        #break
