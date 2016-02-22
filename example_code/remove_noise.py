import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
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

    full_rating = np.genfromtxt('data/rating_matrix.csv', delimiter='\t')[1:, :]
    nan_ind = np.nonzero(np.isnan(full_rating).sum(axis=1))
    full_rating = np.delete(full_rating, nan_ind, axis=0)
    file_ind = full_rating[:, 0]
    full_rating = full_rating[:, 1:]

    feature_arr = np.genfromtxt('data/allFeatures.csv', delimiter=',')[1:, :]
    feature_arr = feature_arr[:, 2:]

    np.savez('tmp/full_rating', file_ind=file_ind, full_rating=full_rating, feature_arr=feature_arr)
    return file_ind, full_rating, feature_arr


def calc_pca(feature_arr, p):
    # Load the face feature matrix
    pca = PCA(n_components=p.explained_variance)
    feature_arr = pca.fit_transform(feature_arr)
    print 'The number of PCs needed to retain {} variance is {}.'.\
        format(p.explained_var, feature_arr.shape[1])


def main():
    p = Params()
    file_ind, full_rating, feature_arr = load_rating_data()
    calc_pca(feature_arr, p)


if __name__ == '__main__':
    main()
