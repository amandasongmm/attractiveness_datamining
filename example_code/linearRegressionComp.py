import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import KFold
np.seterr(divide='ignore', invalid='ignore')

__author__ = 'amanda'


class Params:
    def __init__(self):
        self.explained_var = 0.95
        self.k_fold = 10
        self.model_file_name = 'chicagoFeature'


def load_data():
    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating'].T   # after transpose, the dimension is 200*1540

    # Load the feature data: This part can be replaced into other type of features
    npz_data = np.load('tmp/original_ConfiguralFeatures.npz')  # Chicago feature
    configural_feature_arr = npz_data['feature_arr']
    return configural_feature_arr, orig_rating  # 200 * 29, 200 * 1540.


def plain_linear_regression(orig_feature_arr, full_rating, p):  # feature_arr: 200*7, full_rating: 200* 1540

    # Step 1: Do PCA on orig feature, keep 95% variance
    feature_arr = preprocessing.scale(orig_feature_arr)  # preprocessing
    pca = PCA(n_components=p.explained_var)  # Do pca
    feature_arr = pca.fit_transform(feature_arr)

    '''  Step 2: k-fold train-test split '''
    # Initialize parameters to save.
    total_data_point_num = feature_arr.shape[0]
    rater_num = full_rating.shape[1]

    coef_all = np.zeros((rater_num, feature_arr.shape[1]+1))
    coef_back_all = np.zeros((rater_num, orig_feature_arr.shape[1]))

    # Initialize train and test evaluation metrics
    corr_train, mae_train, mse_train, r_score_train, corr_test, mae_test, mse_test, r_score_test =\
        [np.zeros((rater_num, p.k_fold))] * 8

    # K-fold split
    kf = KFold(total_data_point_num, n_folds=p.k_fold, shuffle=True)
    for i, (train_ind, test_ind) in enumerate(kf):
        print 'K-fold: current i =', i, '\n'
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
        corr_train[:, i], mae_train[:, i], mse_train[:, i], r_score_train[:, i]\
            = linear_model_eval_metrics(y_train, y_train_predict)
        corr_test[:, i], mae_test[:, i], mse_test[:, i], r_score_test[:, i]\
            = linear_model_eval_metrics(y_test, y_test_predict)

    # ensemble model coefficient
    coef_all = coef_all/p.k_fold
    coef_back_all = coef_back_all/p.k_fold

    # training and test metrics
    [corr_train, mae_train, mse_train, r_score_train, corr_test, mae_test, mse_test, r_score_test] =\
        [np.nanmean(x, axis=1) for x in
         [corr_train, mae_train, mse_train, r_score_train, corr_test, mae_test, mse_test, r_score_test]]

    # model_summary = make_dict(corr_train=corr_train, mae_train=mae_train,
    #                           mse_train=mse_train, r_score_train=r_score_train,
    #                           corr_test=corr_test, mae_test=mae_test,
    #                           mse_test=mse_test, r_score_test=r_score_test,
    #                           coef_all=coef_all, coef_back_all=coef_back_all)
    # np.savez('tmp/plain_lr_'+p.model_file_name+'_result', model_summary=model_summary)
    np.savez('tmp/coefficient', coef=coef_all)
    return #model_summary


def make_dict(**kwargs):
    return kwargs


def linear_model_eval_metrics(y_gt, y_predict):

    # Model evaluations: correlation, mean absolute error, mean squared error, r_score
    rater_num = y_gt.shape[1]  # y_gt shape: num_data_point * num_raters
    corr_arr, mae_arr, mse_arr, r_score_arr = [np.zeros((rater_num, 1))] * 4

    for i in range(rater_num):
        corr_arr[i] = np.corrcoef(y_gt[:, i], y_predict[:, i])[0, 1]
        mae_arr[i] = mean_absolute_error(y_gt[:, i], y_predict[:, i])
        mse_arr[i] = mean_squared_error(y_gt[:, i], y_predict[:, i])
        r_score_arr[i] = r2_score(y_gt[:, i], y_predict[:, i])
    [corr_arr, mae_arr, mse_arr, r_score_arr] = [x.flatten() for x in
                                                 [corr_arr, mae_arr, mse_arr, r_score_arr]]
    return corr_arr, mae_arr, mse_arr, r_score_arr


def main():
    p = Params()
    raw_feature_arr, raw_rating = load_data()
    plain_linear_regression(raw_feature_arr, raw_rating, p)
    return


if __name__ == '__main__':
    main()
