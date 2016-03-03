import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import linear_model
from params import Params
import time

__author__ = 'amanda'

'''
The purpose of this code is to generate a clean rating list. Pair twins together.
Filter out adversial rater: who gave the same rating to every face.
If found such a rater, also delete his/ her twin's data. Keep the twin index for later reference.
'''


def clean_rating_data():

    full_rating = np.genfromtxt('data/rating_matrix.csv', delimiter='\t')
    nan_ind = np.nonzero(np.isnan(full_rating).sum(axis=1))
    full_rating = np.delete(full_rating, nan_ind, axis=0)
    file_ind = full_rating[:, 0]
    full_rating = full_rating[:, 1:]

    # Load feature array data, only keep the actual feature arrays.
    feature_arr = np.genfromtxt('data/allFeatures.csv', delimiter=',')[1:, :]
    feature_arr = feature_arr[:, 2:]

    # Save the data for later direct loading.
    np.savez('tmp/test', file_ind=file_ind, full_rating=full_rating, feature_arr=feature_arr)
    return file_ind, full_rating, feature_arr


clean_rating_data()
