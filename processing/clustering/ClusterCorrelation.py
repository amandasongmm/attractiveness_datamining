"""
ClusterCorrelation.py

This script will calculate the average self-correlation for the 3 apparent
clusters seen with analysis of ratings.

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/27/2016
"""

import numpy as np
import pandas as pd

# Load in cluster assignments from Amanda
tmp = np.load('hierarchial_cluster_three_cluster_index.npz')
cluster_ind_list = tmp['cluster_ind_list']

# Load in self-correlation file
corr = pd.read_csv('pearsonVals.csv')

## Correlation and cluster vals are in same order based on previous processing

def avgOfCluster (cNum):
    i = 0
    total = 0
    for x in zip(cluster_ind_list, corr['pears'].tolist()):
        if x[0] == cNum:
            i += 1
            total += x[1]

    return total/i


print 'Cluster 1 self correlation: ' + str(avgOfCluster(1))
print 'Cluster 2 self correlation: ' + str(avgOfCluster(2))
print 'Cluster 3 self correlation: ' + str(avgOfCluster(3))