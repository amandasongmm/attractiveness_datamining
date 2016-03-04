import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
from sklearn.cross_validation import LeaveOneOut
from itertools import chain
import pandas as pd

__author__ = 'amanda'

explained_var = 0.95


def load_data():
    # Load data
    npz_data = np.load('tmp/full_rating.npz')
    full_rating = npz_data['full_rating']
    feature_arr = npz_data['feature_arr']
    return full_rating, feature_arr


def leave_one_out(full_rating, feature_arr):

    # Leave-one-out cross validation
    for cur_dataset_ind in range(4):
        # Prepare feature array
        cur_fea = feature_arr[0+50*cur_dataset_ind: 50+cur_dataset_ind*50, :]
        cur_fea = preprocessing.scale(cur_fea)  # Preprocess the data, standardilzation.
        pca = PCA(n_components=explained_var)
        pca.fit(cur_fea)
        cur_fea = pca.transform(cur_fea)

        # Prepare rating array
        cur_rating = full_rating[:, 0+50*cur_dataset_ind: 50+cur_dataset_ind*50]
        # First, just fit an averaged model.
        ave_rating = np.mean(cur_rating, axis=0)

        loo = LeaveOneOut(len(ave_rating))
        err_array = []
        gt_arry = []
        predict_arr = []
        for train, test in loo:

            # Linear regression
            regr = linear_model.LinearRegression()
            regr.fit(cur_fea[train, :], ave_rating[train])
            rating_predict = regr.predict(cur_fea[test, :])

            # prediction error
            err = np.absolute(rating_predict-ave_rating[test])
            err_array.append(err)
            predict_arr.append(rating_predict)
            gt_arry.append(ave_rating[test])

        t1 = list(chain.from_iterable(predict_arr))
        t2 = list(chain.from_iterable(gt_arry))
        # Correlation in test set
        corr = np.corrcoef(t1, t2)[0, 1]
        print 'correlation: %.2f' % corr
    return


def scatter_plot_4_datasets(full_rating, feature_arr):

    # Prepare a feature dataframe organized by: 'Explained_variance', 'PC Number', 'Dataset Origin'
    dataset_list = ['MIT', 'Glasgow', 'Mixed', 'genHead']
    prediction_gt_dataframe = pd.DataFrame(columns=['Predicted Rating', 'Actual Rating', 'Dataset Origin'])

    # Visualize the datasets with their labels.
    for cur_dataset_ind in range(4):
        # Prepare feature array
        cur_fea = feature_arr[0+50*cur_dataset_ind: 50+cur_dataset_ind*50, :]
        cur_fea = preprocessing.scale(cur_fea)  # Preprocess the data, standardilzation.
        pca = PCA(n_components=explained_var)
        pca.fit(cur_fea)
        cur_fea = pca.transform(cur_fea)

        # Prepare rating array
        cur_rating = full_rating[:, 0+50*cur_dataset_ind: 50+cur_dataset_ind*50]

        # Linear regression
        regr = linear_model.LinearRegression()
        regr.fit(cur_fea, cur_rating)
        rating_predict = regr.predict(cur_fea)
    return


def main():
    full_rating, feature_arr = load_data()
    scatter_plot_4_datasets(full_rating, feature_arr)
    return


if __name__ == '__main__':
    main()
