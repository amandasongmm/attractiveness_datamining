import numpy as np


__author__ = 'amanda'


# Load configural feature array data, only keep the actual feature arrays
orig_feature_arr = np.genfromtxt('../data/allConfiguralFeatures.csv', delimiter=',')[1:, :]
orig_feature_arr = orig_feature_arr[:, 2:]
np.savez('../tmp/original_ConfiguralFeatures', feature_arr=orig_feature_arr)  # Save the data for later direct loading.
