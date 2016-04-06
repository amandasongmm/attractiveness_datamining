import numpy as np

__author__ = 'amanda'


# Load rating data



# First, test on the PCA eigenface feature model.
def load_feature_data1():
    npz_data = np.load('./tmp/pixel_features.npz')
    feature_arr = npz_data['pixel_feature_arr']
    return feature_arr
