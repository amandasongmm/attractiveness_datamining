import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import linear_model
from params import Params
import time

__author__ = 'amanda'


def load_rating_data(gen_new_data=False):
    if not gen_new_data and os.path.exists('tmp/full_rating.npz'):
        npz_data = np.load('tmp/full_rating.npz')
        file_ind = npz_data['file_ind']
        full_rating = npz_data['full_rating']
        feature_arr = npz_data['feature_arr']
        return file_ind, full_rating, feature_arr

    # If it's the first time, run this codes to remove abnormal data entries.
    full_rating = np.genfromtxt('data/rating_matrix.csv', delimiter='\t')[1:, :]
    nan_ind = np.nonzero(np.isnan(full_rating).sum(axis=1))
    full_rating = np.delete(full_rating, nan_ind, axis=0)
    file_ind = full_rating[:, 0]
    full_rating = full_rating[:, 1:]

    # Load feature array data, only keep the actual feature arrays.
    feature_arr = np.genfromtxt('data/allFeatures.csv', delimiter=',')[1:, :]
    feature_arr = feature_arr[:, 2:]

    # Save the data for later direct loading.
    np.savez('tmp/full_rating', file_ind=file_ind, full_rating=full_rating, feature_arr=feature_arr)
    return file_ind, full_rating, feature_arr


def calc_pca(feature_arr, p):

    # Preprocess the data, standardilzation.
    fea_scaled = preprocessing.scale(feature_arr)

    # To confirm if the scaled data has zero mean and unit variance.
    pca = PCA(n_components=p.explained_var)
    feature_arr = pca.fit_transform(fea_scaled)
    print 'The number of PCs needed to retain {} variance is {}.'.\
        format(p.explained_var, feature_arr.shape[1])
    np.savez('tmp/feature_arr', feature_arr=feature_arr)
    return feature_arr


def get_ind_for_pc_dim():
    npz_data = np.load('tmp/feature_arr.npz')
    feature_arr = npz_data['feature_arr']
    pc_dim_total_num = feature_arr.shape[1]

    pc_sorted_ind_list = []
    for cur_dim in range(pc_dim_total_num):
        cur_value_array = feature_arr[cur_dim, :]
        sort_ind_low2high = np.argsort(cur_value_array)
        pc_sorted_ind_list.append(sort_ind_low2high)
    return pc_sorted_ind_list


def main():
    # p = Params()
    # file_ind, full_rating, feature_arr = load_rating_data(True)
    # feature_arr = calc_pca(feature_arr, p)
    get_ind_for_pc_dim()

if __name__ == '__main__':
    main()
