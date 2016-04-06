import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import pearsonr
np.seterr(divide='ignore', invalid='ignore')


__author__ = 'amanda'


# baseline 1: Using aveage score to predict.
def load_data():
    # Load the rating data
    rating_data = np.load('./tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   # after transpose, the dimension is 200*1540
    return orig_rating  # 200 * 29, 200 * 1540.


def average_prediction():
    rating = load_data()
    a = 12
    return rating


average_prediction()