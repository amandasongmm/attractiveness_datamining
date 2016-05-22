import numpy as np
import json
from sklearn import preprocessing

__author__ = 'amanda'


# load the rating matrix
def prepare_rating_data():

    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   # 1540 * 200

    # Z-score to make everyone's ratings have zeros mean and unit std.
    scaler = preprocessing.StandardScaler().fit(orig_rating.T)  # it only fit to the second dim. So flap it first.
    dataset_mean = scaler.mean_
    dataset_std = scaler.scale_

    z_scored_rating = scaler.transform(orig_rating.T)
    z_scored_rating = z_scored_rating.T

    # randomly permute the rating matrix
    permuted_rating = z_scored_rating[:, np.random.permutation(z_scored_rating.shape[1])]

    # Compute average rating. Do z-score
    averaged_rating = z_scored_rating.mean(axis=0)
    ave_mean = averaged_rating.mean()
    ave_std = averaged_rating.std()
    z_scored_average_rating = (averaged_rating-ave_mean)/ave_std

    return z_scored_rating, dataset_mean, dataset_std, permuted_rating, z_scored_average_rating, ave_mean, ave_std


def prepare_feature():



def linear_regression():




