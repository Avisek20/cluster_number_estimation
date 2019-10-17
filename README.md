# Estimate the number of clusters using

# The Last Leap and The Last Major Leap

Paper Source: [Gupta A., Datta S., Das S., "Fast automatic estimation of the
number of clusters from the minimum inter-center distance for k-means
clustering", Pattern Recognition Letters, vol. 116, pp. 72-79, 2018.](https://www.sciencedirect.com/science/article/pii/S0167865518305579)

* The Last Leap prefers identification of separate clusters.

![last_leap_example](https://raw.githubusercontent.com/Avisek20/cluster_number_estimation/master/imgs/diff_scale_ll.png)

* The Last Major Leap prefers identification of equal-sized clusters.

![last_major_leap_example](https://raw.githubusercontent.com/Avisek20/cluster_number_estimation/master/imgs/diff_scale_lml.png)


A comparison of the performance of both methods -

* The Last Leap -

![last_leap_performance](https://raw.githubusercontent.com/Avisek20/cluster_number_estimation/master/imgs/last_leap.png)

* The Last Major Leap

![last_major_leap_performance](https://raw.githubusercontent.com/Avisek20/cluster_number_estimation/master/imgs/last_major_leap.png)

## Comparative Experiments

The methods of the Last Leap and the Last Major Leap are compared with several popularly used cluster number estimation methods and indices. The experiment files corresponding to the source paper are present in the **experiments** directory.

