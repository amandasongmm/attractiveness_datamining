import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import train_test_split
import bokeh.plotting as plt
import ujson
from scipy.stats import pearsonr


__author__ = 'amanda'


def load_data():
    # Load the rating data
    rating_data = np.load('./tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   # after transpose, the dimension is 1540 * 200.

    with open('./data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)
    return orig_rating, image_list


def beauty_select():
    orig_rating, image_list = load_data()
    rating_mean = np.mean(orig_rating, axis=0)
    rating_sort_ind = np.argsort(rating_mean)

    rating_std = np.std(orig_rating, axis=0)
    std_sort_ind = np.argsort(rating_std)

    high_score = rating_mean[rating_sort_ind[-1]]
    high_name = image_list[rating_sort_ind[-1]]

    low_score = rating_mean[rating_sort_ind[0]]
    low_name = image_list[rating_sort_ind[0]]

    print high_name, low_name
    return

beauty_select()
