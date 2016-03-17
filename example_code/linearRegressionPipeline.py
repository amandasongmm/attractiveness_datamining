import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import KFold
from bokeh.charts import Bar, output_file, show


__author__ = 'amanda'


class Params:
    def __init__(self):
        self.explained_var = 0.95
        self.k_fold = 10


def load_data():
    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']
    orig_rating = orig_rating.T  # after transpose, the dimension is 200*1540

    # Load the feature data: This part can be replaced into other type of features
    npz_data = np.load('tmp/original_features.npz')
    configural_feature_arr = npz_data['feature_arr']
    return configural_feature_arr, orig_rating  # 200 * 29, 200 * 1540.


def make_dict(**kwargs):
    return kwargs


def linear_model_eval_metrics(y_gt, y_predict):

    # Model evaluations: correlation, mean absolute error, mean squared error, r_score
    rater_num = y_gt.shape[1]  # y_gt shape: num_data_point * num_raters
    corr_arr = np.zeros((rater_num, 1))
    mae_arr = np.zeros((rater_num, 1))
    mse_arr = np.zeros((rater_num, 1))
    r_score_arr = np.zeros((rater_num, 1))

    for i in range(rater_num):
        corr_arr[i] = np.corrcoef(y_gt[:, i], y_predict[:, i])[0, 1]
        mae_arr[i] = mean_absolute_error(y_gt[:, i], y_predict[:, i])
        mse_arr[i] = mean_squared_error(y_gt[:, i], y_predict[:, i])
        r_score_arr[i] = r2_score(y_gt[:, i], y_predict[:, i])
    return corr_arr, mae_arr, mse_arr, r_score_arr


def plain_linear_regression(feature_arr, full_rating, p):  # feature_arr: 200*7, full_rating: 200* 1540

    ''' Step 1: Do PCA on orig feature, keep 95% variance '''
    explained_var = 0.95
    feature_arr = preprocessing.scale(feature_arr)  # preprocessing
    pca = PCA(n_components=explained_var)  # Do pca
    feature_arr = pca.fit_transform(feature_arr)

    '''  Step 2: k-fold train-test split '''
    # Initialize parameters to save.
    total_data_point_num = feature_arr.shape[0]
    rater_num = full_rating.shape[1]

    coef_all = np.zeros((rater_num, feature_arr.shape[1]+1))
    coef_back_all = np.zeros((rater_num, feature_arr.shape[1]))

    # train evaluation metrics
    corr_all_train = np.zeros((rater_num, p.k_fold))
    mae_all_train = np.zeros((rater_num, p.k_fold))
    mse_all_train = np.zeros((rater_num, p.k_fold))
    r_score_all_train = np.zeros((rater_num, p.k_fold))

    # Test evaluation metrics
    corr_all_test = np.zeros((rater_num, p.k_fold))
    mae_all_test = np.zeros((rater_num, p.k_fold))
    mse_all_test = np.zeros((rater_num, p.k_fold))
    r_score_all_test = np.zeros((rater_num, p.k_fold))

    # K-fold split
    kf = KFold(total_data_point_num, n_folds=p.k_fold, shuffle=True)
    i = 0
    for train_ind, test_ind in kf:
        # Generate training, test datasets
        x_train, x_test = feature_arr[train_ind, :], feature_arr[test_ind, :]
        y_train, y_test = full_rating[train_ind, :], full_rating[test_ind, :]

        # Basic model
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)

        # model coefficient params
        coef_full = np.column_stack((regr.coef_, regr.intercept_))  # 1540 * 8
        coef_in_rawspace = pca.inverse_transform(regr.coef_)  # before: 1540*7, after: 1540 * 29
        coef_all = coef_all + coef_full
        coef_back_all = coef_back_all + coef_in_rawspace

        # Model prediction
        y_train_predict = regr.predict(x_train)
        y_test_predict = regr.predict(x_test)

        # Model evaluation metrics.
        corr_all_train[:, i], mae_all_train[:, i], mse_all_train[:, i], r_score_all_train[:, i]\
            = linear_model_eval_metrics(y_train, y_train_predict)
        corr_all_test[:, i], mae_all_test[:, i], mse_all_test[:, i], r_score_all_test[:, i]\
            = linear_model_eval_metrics(y_test, y_test_predict)
        i += 1

    # ensemble model coefficient
    coef_full = coef_full/p.k_fold
    coef_back_all = coef_back_all/p.k_fold

    # training metrics
    corr_all_train = np.nanmean(corr_all_train, axis=1)
    mae_all_train = np.nanmean(mae_all_train, axis=1)
    mse_all_train = np.nanmean(mse_all_train, axis=1)
    r_score_all_train = np.nanmean(r_score_all_train, axis=1)

    # testing metrics
    corr_all_test = np.nanmean(corr_all_test, axis=1)
    mae_all_test = np.nanmean(mae_all_test, axis=1)
    mse_all_test = np.nanmean(mse_all_test, axis=1)
    r_score_all_test = np.nanmean(r_score_all_test, axis=1)

    metric_summary = make_dict(corr_all_train=corr_all_train, mae_all_train=mae_all_train,
                               mse_all_train=mse_all_train, r_score_all_train=r_score_all_train,
                               corr_all_test=corr_all_test, mae_all_test=mae_all_test,
                               mse_all_test=mse_all_test, r_score_all_test=r_score_all_test)

    return coef_full, coef_back_all, metric_summary


def visualize_linear_model_metrics(data_summary):
    data = {'source': ['training', 'training', 'training', 'training',
                       'testing', 'testing', 'testing', 'testing'],
            'metric_name': ['correlation', 'MAE', 'MSE', 'r2score',
                            'correlation', 'MAE', 'MSE', 'r2score'],
            'value': [np.mean(x) for x in data_summary],
            'data_summary': data_summary}

    bar = Bar(data, values='value', label='metric_name', group='source',
              title='Linear Regression Model Performance', legend='top_right')
    output_file("bar.html")
    show(bar)
    return


def visualize_coef_bar(coef_full):

    # coef_full = coef_full[:, 0:-1]
    num_features = coef_full.shape[1]  # number of data entries
    x = np.arange(num_features)  # the x locations for the groups

    # Prepare data
    y_mean = np.mean(coef_full, axis=0)
    y_std = np.mean(coef_full, axis=0)
    width = 0.5  # bar width

    # Plot
    fig, ax = plt.subplots()
    ax.bar(x, y_mean, width,  color='MediumSlateBlue', yerr=y_std,  error_kw={'ecolor': 'Tomato', 'linewidth': 2})
    axes = plt.gca()
    axes.set_ylim([-0.5, 0.5])
    ax.set_xticks(x+width)
    # ax.set_xticklabels(('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'))
    plt.show()

    return


def main():
    p = Params()
    raw_feature_arr, raw_rating = load_data()
    return


if __name__ == '__main__':
    main()


