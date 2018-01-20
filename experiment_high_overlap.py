import os
import time
import datetime
import numpy as np
from numpy.random import normal
from scipy.spatial.distance import cdist
from indices.bensaid_hall_bezdek.bensaid_hall_bezdek import bensaid_hall_bezdek
from indices.calinski_harabasz.calinski_harabasz import calinski_harabasz
from indices.classification_entropy81.classification_entropy81\
    import classification_entropy81
from indices.davies_bouldin.davies_bouldin import davies_bouldin
from indices.dunn73.dunn73 import dunn73
from indices.fukuyama_sugeno89.fukuyama_sugeno89 import fukuyama_sugeno89
from indices.fuzzy_hypervolume89.fuzzy_hypervolume89 import fuzzy_hypervolume89
from indices.gap_statistic.gap_statistic2 import gap_statistic
from indices.halkidi_vazirgannis.halkidi_vazirgannis import halkidi_vazirgannis
from indices.hartigan85.hartigan85 import hartigan85
from indices.I_index.I_index import I_index
from indices.jump_method.jump_method import jump_method
from indices.krzanowski_lai.krzanowski_lai2 import krzanowski_lai
from indices.modified_partition_coefficient96.modified_partition_coefficient96\
    import modified_partition_coefficient96
from indices.partition_coefficient74.partition_coefficient74\
    import partition_coefficient74
from indices.pbmf.pbmf import pbmf
from indices.pcaes.pcaes import pcaes
from indices.prediction_strength.prediction_strength import run_prediction_strength
from indices.ratkowsky_lance.ratkowsky_lance import ratkowsky_lance
from indices.rezaee10.rezaee10 import rezaee10
from indices.rezaee_lelieveldt_reiber98.rezaee_lelieveldt_reiber98\
    import rezaee_lelieveldt_reiber98
from indices.slope_statistic.slope_statistic import slope_statistic
from indices.xie_beni.xie_beni import xie_beni
from indices.xu_index.xu_index2 import xu_index
from indices.zhao_xu_franti09.zhao_xu_franti09 import zhao_xu_franti09
from indices.min_dist_method.min_dist_method import LastLeap
from indices.min_dist_method.min_dist_method import LastMajorLeap


def run_bensaid_hall_bezdek(data, all_centers, all_labels) :
    return bensaid_hall_bezdek(data, all_centers, all_labels, m=2)

def run_calinski_harabasz(data, all_centers, all_labels) :
    return calinski_harabasz(data, all_centers, all_labels)

def run_classification_entropy81(data, all_centers, all_labels) :
    return classification_entropy81(data, all_centers, all_labels)

def run_davies_bouldin(data, all_centers, all_labels) :
    return davies_bouldin(data, all_centers, all_labels)

def run_dunn73(data, all_centers, all_labels) :
    return dunn73(data, all_centers, all_labels)

def run_fukuyama_sugeno89(data, all_centers, all_labels) :
    return fukuyama_sugeno89(data, all_centers, all_labels)

def run_fuzzy_hypervolume89(data, all_centers, all_labels) :
    return fuzzy_hypervolume89(data, all_centers, all_labels)

def run_gap_statistic(data, all_centers, all_labels) :
    estimated_k, logWk, logWk_ref, estimated_k_nolog, Wk, Wk_ref = \
        gap_statistic(data, all_centers, all_labels, B=30, method='pca')
    return (estimated_k, estimated_k_nolog), (logWk, logWk_ref, Wk, Wk_ref)

def run_halkidi_vazirgannis(data, all_centers, all_labels) :
    return halkidi_vazirgannis(data, all_centers, all_labels)

def run_hartigan85(data, all_centers, all_labels) :
    return hartigan85(data, all_centers, all_labels)

def run_I_index(data, all_centers, all_labels) :
    return I_index(data, all_centers, all_labels)

def run_jump_method(data, all_centers, all_labels) :
    return jump_method(data, all_centers, all_labels)

def run_krzanowski_lai(data, all_centers, all_labels) :
    return krzanowski_lai(data, all_centers, all_labels)

def run_modified_partition_coefficient96(data, all_centers, all_labels) :
    return modified_partition_coefficient96(data, all_centers, all_labels)

def run_partition_coefficient74(data, all_centers, all_labels) :
    return partition_coefficient74(data, all_centers, all_labels)

def run_pbmf(data, all_centers, all_labels) :
    return pbmf(data, all_centers, all_labels)

def run_pcaes(data, all_centers, all_labels) :
    return pcaes(data, all_centers, all_labels)

def run_prediction_strength2(data, all_centers, all_labels) :
    return run_prediction_strength(data)

def run_ratkowsky_lance(data, all_centers, all_labels) :
    return ratkowsky_lance(data, all_centers, all_labels)

def run_rezaee10(data, all_centers, all_labels) :
    return rezaee10(data, all_centers, all_labels)

def run_rezaee_lelieveldt_reiber98(data, all_centers, all_labels) :
    return rezaee_lelieveldt_reiber98(data, all_centers, all_labels)

def run_slope_statistic(data, all_centers, all_labels) :
    return slope_statistic(data, all_centers, all_labels)

def run_xie_beni(data, all_centers, all_labels) :
    return xie_beni(data, all_centers, all_labels)

def run_xu_index(data, all_centers, all_labels) :
    return xu_index(data, all_centers, all_labels)

def run_zhao_xu_franti09(data, all_centers, all_labels) :
    return zhao_xu_franti09(data, all_centers, all_labels)

def run_LastLeap(all_centers, all_labels) :
    return LastLeap(all_centers, all_labels)

def run_LastMajorLeap(all_centers, all_labels) :
    return LastMajorLeap(all_centers, all_labels)

if __name__ == '__main__' :
    # -------------------------------------------------------------------
    # For each index set options and results_directory with associated values
    options = {
        0 : run_LastLeap,
        1 : run_LastMajorLeap,
        2 : run_bensaid_hall_bezdek,
        3 : run_calinski_harabasz,
        4 : run_classification_entropy81,
        5 : run_davies_bouldin,
        6 : run_dunn73,
        7 : run_fukuyama_sugeno89,
        8 : run_fuzzy_hypervolume89,
        9 : run_gap_statistic,
        10 : run_halkidi_vazirgannis,
        11 : run_hartigan85,
        12 : run_I_index,
        13 : run_jump_method,
        14 : run_krzanowski_lai,
        15 : run_modified_partition_coefficient96,
        16 : run_partition_coefficient74,
        17 : run_pbmf,
        18 : run_pcaes,
        19 : run_prediction_strength2,
        20 : run_rezaee10,
        21 : run_rezaee_lelieveldt_reiber98,
        22 : run_slope_statistic,
        23 : run_xie_beni,
        24 : run_xu_index,
        25 : run_zhao_xu_franti09,
        }

    results_directory = {
        0 : 'LastLeap',
        1 : 'LastMajorLeap',
        2 : 'bensaid_hall_bezdek',
        3 : 'calinski_harabasz',
        4 : 'classification_entropy81',
        5 : 'davies_bouldin',
        6 : 'dunn73',
        7 : 'fukuyama_sugeno89',
        8 : 'fuzzy_hypervolume89',
        9 : 'gap_statistic',
        10 : 'halkidi_vazirgannis',
        11 : 'hartigan85',
        12 : 'I_index',
        13 : 'jump_method',
        14 : 'krzanowski_lai',
        15 : 'modified_partition_coefficient96',
        16 : 'partition_coefficient74',
        17 : 'pbmf',
        18 : 'pcaes',
        19 : 'prediction_strength',
        20 : 'rezaee10',
        21 : 'rezaee_lelieveldt_reiber98',
        22 : 'slope_statistic',
        23 : 'xie_beni',
        24 : 'xu_index',
        25 : 'zhao_xu_franti09',
        }

    # ------------------------------------------------------------------- #

    dataset_dir = 'data_high_overlap'
    centers_dir = 'cluster_'+dataset_dir
    d1 = datetime.date
    results_today = 'results-'+d1.today().isoformat()+'_'+dataset_dir

    # ------------------------------------------------------------------- #

    list_indices = np.range(26)


    for SET_INDEX in list_indices:

        if not os.path.exists(results_today+'/'+results_directory[SET_INDEX]) :
            os.makedirs(results_today+'/'+results_directory[SET_INDEX])

        for dir_num in range(100) :

            fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                '/result'+str(dir_num)+'.txt', 'w')
            fw.close()
            fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                '/time'+str(dir_num)+'.txt', 'w')
            fw.close()
            if SET_INDEX == 29 : # Slope and sillhouette
                if not os.path.exists(results_today+'/sillhouette') :
                    os.makedirs(results_today+'/sillhouette')
                fw2 = open(results_today+'/sillhouette/result'+str(dir_num)+'.txt', 'w')
                fw2.close()
                fw2 = open(results_today+'/sillhouette/time'+str(dir_num)+'.txt', 'w')
                fw2.close()
            if SET_INDEX == 12 : # Gap-Stat and Mohajer Gap-Stat
                if not os.path.exists(results_today+'/gap_mohajer') :
                    os.makedirs(results_today+'/gap_mohajer')
                fw2 = open(results_today+'/gap_mohajer/result'+str(dir_num)+'.txt', 'w')
                fw2.close()
                fw = open(results_today+'/gap_mohajer/result'+str(dir_num)+'.txt', 'w')
                fw.close()

            for a_dataset in range(5) :
                print(results_directory[SET_INDEX], dir_num, a_dataset)
                #print('data',a_dataset)

                # Load data
                data = \
                np.loadtxt(dataset_dir+'/'+dataset_dir+'_'+str(dir_num)+'/data_'+str(a_dataset)+'.txt')
                #label = np.array(data[:,-1])
                data = np.array(data[:,0:-1])
                # sqrt{n} clusters are considered
                maxK = int(np.ceil(data.shape[0]**0.5))

                # Load centers and labels
                all_centers = []
                all_labels = []
                for k in range(1,maxK+1) :
                    # Load centers
                    centers = np.loadtxt(centers_dir+'/'+dataset_dir+'_'+str(dir_num)+ \
                        '/data_'+str(a_dataset)+'_k_'+str(k)+'_centers.txt')
                    if len(centers.shape) == 1:
                        centers = np.reshape(centers, (1,centers.shape[0]))
                    all_centers.append(centers)
                    # Load labels
                    #labels = \
                    # np.loadtxt(centers_dir+'/'+dataset_dir+'_'+str(dir_num)+ \
                    #    '/data_'+str(a_dataset)+'_k_'+str(k)+'_labels.txt')
                    labels = np.argmin(cdist(data, centers), axis=1)
                    all_labels.append(labels)

                # Run the index, get estimated k
                start = time.time()
                est_k, index_values = options[SET_INDEX](data, all_centers, all_labels)
                end = time.time() - start
                #print(est_k)

                # Write estimated k to file
                if SET_INDEX not in [12, 29] :
                    fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                        '/result'+str(dir_num)+'.txt', 'a')
                    fw.write(str(est_k)+'\n')
                    fw.close()
                    fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                        '/time'+str(dir_num)+'.txt', 'a')
                    fw.write(str(end)+'\n')
                    fw.close()
                elif SET_INDEX == 12 : # Gapstat and Mohajer gap
                    fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                    '/result'+str(dir_num)+'.txt', 'a')
                    fw.write(str(est_k[0])+'\n')
                    fw.close()
                    fw2 = open(results_today+'/gap_mohajer/result'+str(dir_num)+'.txt', 'a')
                    fw2.write(str(est_k[1])+'\n')
                    fw2.close()
                    fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                        '/time'+str(dir_num)+'.txt', 'a')
                    fw.write(str(end)+'\n')
                    fw.close()
                    fw2 = open(results_today+'/gap_mohajer/time'+str(dir_num)+'.txt', 'a')
                    fw2.write(str(end)+'\n')
                    fw2.close()
                elif SET_INDEX == 29 : # Slope and sillhouette
                    fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                    '/result'+str(dir_num)+'.txt', 'a')
                    fw.write(str(est_k[0])+'\n')
                    fw.close()
                    fw2 = open(results_today+'/sillhouette/result'+str(dir_num)+'.txt', 'a')
                    fw2.write(str(est_k[1])+'\n')
                    fw2.close()
                    fw = open(results_today+'/'+results_directory[SET_INDEX]+\
                        '/time'+str(dir_num)+'.txt', 'a')
                    fw.write(str(end)+'\n')
                    fw.close()
                    fw2 = open(results_today+'/sillhouette/time'+str(dir_num)+'.txt', 'a')
                    fw2.write(str(end)+'\n')
                    fw2.close()

                #if a_dataset == 'examples7_v2.txt' :
                #   break
