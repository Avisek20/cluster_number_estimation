import numpy as np

data_folders = [
    'data_well_separated', 'data_diff_spread',
    'data_slightly_diff_points', 'data_very_diff_points',
    'data_slight_overlap2', 'data_high_overlap2'
]
n_data_dirs = [30, 30, 30, 30, 30, 30]
n_data_sets = [25, 25, 25, 25, 5, 5]

for iter_data in range(len(data_folders)):
    directory = 'results_km_'+data_folders[iter_data]
    results_arr = np.zeros((
        n_data_dirs[iter_data], n_data_sets[iter_data], 30
    ))
    for iter2 in range(n_data_dirs[iter_data]):
        for iter3 in range(n_data_sets[iter_data]):
            results_arr[iter2,iter3,:] = np.loadtxt(
                directory+'/'+data_folders[iter_data]+'_'+str(iter2)+'/'+'results_'+str(iter3)
            )
    np.savetxt(
        'output/results_'+data_folders[iter_data]+'_all.txt',
        results_arr.reshape(
            n_data_dirs[iter_data] * n_data_sets[iter_data], 30
        ), fmt='%d'
    )
