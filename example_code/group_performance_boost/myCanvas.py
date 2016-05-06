import numpy as np
from sklearn import preprocessing

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn import cross_validation

__author__ = 'amanda'


def preprocess_data():
    rating = np.load('./data_ready_to_use/clean_rating_data.npz')
    rating = rating['full_rating']
    average_rating = rating.mean(axis=0)

    feature_arr = np.load('./data_ready_to_use/clean_features.npz')
    feature_arr = feature_arr['feature_arr']

    gender_list = np.load('./data_ready_to_use/gender_list.npz')
    gender_list = gender_list['gender_list']

    gender_list = gender_list[:, np.newaxis]
    feature_arr = np.hstack((gender_list, feature_arr))
    feature_arr = preprocessing.scale(feature_arr)
    return average_rating, feature_arr, gender_list

# Load the preprocessed data.
rating, feature_arr = preprocess_data()


cv = cross_validation.ShuffleSplit(5, n_iter=10, test_size=0.15, random_state=0)


# Fit estimators
ESTIMATORS = {"Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
              "K-nn": KNeighborsRegressor(),
              "Linear regression": LinearRegression(),
              "Ridge": RidgeCV()}

y_test_predict = dict()
y_overall_predict = dict()
for name, estimator in ESTIMATORS.items():
    cross_validation.cross_val_score(estimator, feature_arr, rating, cv=cv)
