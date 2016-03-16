import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from bokeh.charts import output_file, show, Histogram

__author__ = 'amanda'


def load_data():
    # Load the data
    rating_data = np.load('tmp/clean_rating_data.npz')
    full_rating = rating_data['full_rating']
    feature_data = np.load('tmp/feature_arr.npz')
    feature_arr = feature_data['feature_arr']
    full_rating = full_rating.T  # after transpose, the dimension is 200*1540
    # do preprocessing to the feature array.
    # feature_arr = preprocessing.scale(feature_arr)  # mean in each feature dim = 0, std =1.
    return feature_arr, full_rating  # 200 * 29, 200 * 1540.


# def pca_feature(feature_array, preprocess=1):  # do the preprocessing, unless otherwise specified
#     explained_var = 0.95
#     if preprocess == 1:
#         feature_array = preprocessing.scale(feature_array)
#
#     # To confirm if the scaled data has zero mean and unit variance.
#     pca = PCA(n_components=explained_var)
#     feature_array = pca.fit_transform(feature_array)
#
#     print 'The number of PCs needed to retain {} variance is {}.\n PCA processing done...\n\n'.\
#         format(explained_var, feature_array.shape[1])
#     return feature_array


def plain_lg_all_sub(feature_arr, full_rating):  # feature_arr: 200*7, full_rating: 200* 1540

    # combine doing PCA into the pipeline
    explained_var = 0.95
    feature_arr = preprocessing.scale(feature_arr)  # preprocessing
    pca = PCA(n_components=explained_var)  # Do pca
    feature_arr = pca.fit_transform(feature_arr)

    # Basic model
    regr = linear_model.LinearRegression()
    regr.fit(feature_arr, full_rating)

    # Model prediction
    rating_predict = regr.predict(feature_arr)

    # model params
    coef = regr.coef_
    intercept = regr.intercept_
    coefficient_full = np.column_stack((coef, intercept))
    coef_back_in_rawspace = pca.inverse_transform(coef)  # before: 1540*7, after: 1540 * 29

    # Model evaluations: mean absolute error, mean squared error, r_score
    MAE = mean_absolute_error(full_rating, rating_predict)
    MSE = mean_squared_error(full_rating, rating_predict)
    r_score = r2_score(full_rating, rating_predict)

    # Compute the correlation
    rater_num = full_rating.shape[1]
    correlation_array = np.zeros((rater_num, 1))
    for i in range(rater_num):
        correlation_array[i] = np.corrcoef(rating_predict[:, i], full_rating[:, i])[0, 1]
    correlation_array = [item for sublist in correlation_array for item in sublist]      # Flatten the list

    # Print the result
    print 'The group average corcoe is {:.2f}, the std is {:.2f}\n ' \
          'MAE={:.2f}, MSE={:.2f}, r_score={:.2f}'\
        .format(np.mean(correlation_array), np.std(correlation_array), MAE, MSE, r_score)
    return correlation_array, coefficient_full, coef_back_in_rawspace


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

feature_arr, full_rating = load_data()
# feature_arr = pca_feature(feature_arr)
ccorrelation_array, coefficient_full, coef_back_in_rawspace = plain_lg_all_sub(feature_arr, full_rating)
visualize_coef_bar(coef_back_in_rawspace)

