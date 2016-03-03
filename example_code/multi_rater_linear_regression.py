import numpy as np
from sklearn import preprocessing, linear_model
# from sklearn.decomposition import PCA
# from sklearn.cross_validation import LeaveOneOut
# from itertools import chain
# import pandas as pd
from bokeh.charts import output_file, show, Histogram
from sklearn.cross_validation import train_test_split


__author__ = 'amanda'


def load_data():
    # Load the data
    npz_data = np.load('tmp/full_rating.npz')
    full_rating = npz_data['full_rating']
    feature_arr = npz_data['feature_arr']
    full_rating = full_rating.T  # after transpose, the dimension is 200*1545

    # Check if anyone gives the same rating to all faces. If so, delete him.
    assole_sub_ind = [i for i, x in enumerate(np.std(full_rating, axis=0)) if x == 0]
    full_rating = np.delete(full_rating, assole_sub_ind, axis=1)
    return feature_arr, full_rating


'''
Task 1.
Compute the prediction's correlation with every rater using the linear model's prediction.
No preprocessing of the data.
'''


def plain_lg_all_sub(feature_arr, full_rating):
    regr = linear_model.LinearRegression()
    regr.fit(feature_arr, full_rating)
    rating_predict = regr.predict(feature_arr)
    rater_num = full_rating.shape[1]
    correlation_list = np.zeros((rater_num, 1))

    # Compute the correlation
    for i in range(rater_num):
        correlation_list[i] = np.corrcoef(rating_predict[:, i], full_rating[:, i])[0, 1]

    # Flatten the list
    correlation_list = [item for sublist in correlation_list for item in sublist]
    cor_mean = np.mean(correlation_list)
    cor_std = np.std(correlation_list)
    print_string = 'The group average corcoe is {:.2f}, the std is {:.2f}'.format(cor_mean, cor_std)
    print print_string
    return correlation_list, cor_mean, cor_std

'''
Task 2. Calculate the average rating for each face. Fit a model. Compute the correlation.
Purpose: To see if the average rating is easier to predict compared with each individual.
'''


def plain_lg_average(feature_arr, full_rating):  # feature_arr 200*29, full_rating 200 * 1544
    regr = linear_model.LinearRegression()
    average_rating = np.mean(full_rating, axis=1)
    regr.fit(feature_arr, average_rating)
    rating_predict = regr.predict(feature_arr)
    corr_coef = np.corrcoef(rating_predict, average_rating)[0, 1]

    print_string = 'The correlation coefficient of average attractiveness rating and the prediction ' \
                   'is {:.2f}'.format(corr_coef)
    print print_string
    return corr_coef


def visualize_corrcoef_hist(correlation_list, output_file_name):
    # Visualize the histogram of correlation coefficient for all subjects.
    p = Histogram(correlation_list, title="Correlation Coefficient")
    output_file(output_file_name)
    show(p)
    return


def cross_val_lg_all_sub(feature_arr, full_rating):
    correlation_list_acc = []
    itr_num = 10  # Do cross validation for itr_num of times. Then plot all the correlations
    for cur_itr in range(itr_num):
        x_train, x_test, y_train, y_test = train_test_split(feature_arr, full_rating, test_size=0.1)
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        rating_predict = regr.predict(x_test)
        rater_num = full_rating.shape[1]
        correlation_list = np.zeros((rater_num, 1))

        # Compute the correlation
        for i in range(rater_num):
            correlation_list[i] = np.corrcoef(rating_predict[:, i], y_test[:, i])[0, 1]

        # Flatten the list
        correlation_list = [item for sublist in correlation_list for item in sublist]
        correlation_list_acc = correlation_list + correlation_list_acc

    # correlation_list = [x for x in correlation_list_acc]

    # Delete the NAN entries.
    isnan_index = np.nonzero(np.isnan(correlation_list_acc))
    correlation_list_acc = np.delete(correlation_list_acc, isnan_index)
    cor_mean = np.mean(correlation_list_acc)
    cor_std = np.std(correlation_list_acc)
    print_string = 'The group average corcoe is {:.2f}, the std is {:.2f}'.format(cor_mean, cor_std)
    print print_string
    return correlation_list_acc, cor_mean, cor_std


def main():
    feature_arr, full_rating = load_data()

    # correlation_list1 = plain_lg_all_sub(feature_arr, full_rating)
    # visualize_corrcoef_hist(correlation_list1, "./linear_reg_figs/popuPredictionCorrHist1.html")

    correlation_list2 = cross_val_lg_all_sub(feature_arr, full_rating)
    visualize_corrcoef_hist(correlation_list2, "./linear_reg_figs/popuPredictionCorrHist2_withCrosVal.html")
    return correlation_list2


if __name__ == '__main__':
    correlation_list = main()
    a = 12

