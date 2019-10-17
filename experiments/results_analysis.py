import numpy as np

data_folders = [
    'data_well_separated', 'data_diff_spread',
    'data_slightly_diff_points', 'data_very_diff_points',
    'data_slight_overlap2', 'data_high_overlap2'
]

for iter_data in range(len(data_folders)):
    results = np.loadtxt(
        'output/results_'+data_folders[iter_data]+'_all.txt',
    )
    key = np.loadtxt('keys/key_'+data_folders[iter_data][5:]+'.txt')
    if key.ndim == 1:
        key = key[:,None]
    if iter_data < 4:
        final_results = (results == key).sum(axis=0)
        print(final_results)
        print(results.shape[0],final_results[15:16+1],(final_results[15:16+1]*100)/results.shape[0])
        print('-----------------')
    else:
        final_results = (np.abs(results - key[:,0][:,None])<=2).sum(axis=0)
        print(final_results)
        print(results.shape[0],final_results[15:16+1],(final_results[15:16+1]*100)/results.shape[0])
        print('-------')
        final_results = (np.abs(results - key[:,1][:,None])<=2).sum(axis=0)
        print(final_results)
        print(results.shape[0],final_results[15:16+1],(final_results[15:16+1]*100)/results.shape[0])
        print('-----------------')
