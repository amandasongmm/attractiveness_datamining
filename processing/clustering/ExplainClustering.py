"""
ExplainClustering.py

This script will calculate the gender ratio and average age of each cluster seen with analysis of ratings.

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/27/2016
"""

import numpy as np
import pandas as pd

# Load in cluster assignments from Amanda
tmp = np.load('hierarchial_cluster_three_cluster_index.npz')
cluster_ind_list = tmp['cluster_ind_list']

# Load in bio data file
bio = pd.read_csv('bioData.csv')

# Remove bad indexes from Amanda
for x in [90, 93, 337, 417]:
    bio = bio[bio.pair_nums != x]

## Correlation and cluster vals are in same order based on previous processing

def ratioOfCluster (cNum):
    two = 0
    one = 0
    for x in zip(cluster_ind_list, bio['gender'].tolist()):
        if x[0] == cNum:
            if x[1] == 1:
                one += 1
            else:
                two += 1
    return (float(one)/two)

def averageOfCluster (cNum):
    i = 0
    total = 0
    for x in zip(cluster_ind_list, bio['age'].tolist()):
        if x[0] == cNum:
            i += 1
            total += x[1]

    return float(total)/i


print 'Cluster 1 gender ratio male/female: ' + str(ratioOfCluster(1))
print 'Cluster 2 gender ratio male/female: ' + str(ratioOfCluster(2))
print 'Cluster 3 gender ratio male/female: ' + str(ratioOfCluster(3))
print '----------'
print 'Cluster 1 average age: ' + str(averageOfCluster(1))
print 'Cluster 2 average age: ' + str(averageOfCluster(2))
print 'Cluster 3 average age: ' + str(averageOfCluster(3))