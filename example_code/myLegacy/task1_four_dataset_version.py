import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

__author__ = 'amanda'


def main():
    # Load the data
    rating_data = np.load('tmp/clean_rating_data.npz')
    full_rating = rating_data['full_rating']
    feature_data = np.load('tmp/feature_arr.npz')
    raw_feature_arr = feature_data['feature_arr']
    full_rating = full_rating.T  # after transpose, the dimension is 200*1540
    # 200 * 29, 200 * 1540.
    dataset_list = ['MIT', 'Glasgow', 'Mixed', 'genHead']

    for cur_dataset_ind in range(4):
        # Prepare feature array
        cur_fea = raw_feature_arr[0+50*cur_dataset_ind: 50+cur_dataset_ind*50, :]
        cur_rating = full_rating[0+50*cur_dataset_ind: 50+cur_dataset_ind*50, :]

        # Do PCA
        explained_var = 0.95
        feature_arr = preprocessing.scale(cur_fea)  # preprocessing
        pca = PCA(n_components=explained_var)  # Do pca
        feature_arr = pca.fit_transform(feature_arr)
        pc_dim_num = feature_arr.shape[1]
        print 'Dataset: {} The number of PCs needed to retain {} variance is {}.\n'\
            .format(dataset_list[cur_dataset_ind], explained_var, pc_dim_num)

        # Basic model
        regr = linear_model.LinearRegression()
        regr.fit(feature_arr, cur_rating)

        # Model prediction
        rating_predict = regr.predict(feature_arr)

        # model params
        coef = regr.coef_
        intercept = regr.intercept_
        coefficient_full = np.column_stack((coef, intercept))
        coef_back_in_rawspace = pca.inverse_transform(coef)  # before: 1540*7, after: 1540 * 29

        # Model evaluations: mean absolute error, mean squared error, r_score
        MAE = mean_absolute_error(cur_rating, rating_predict)
        MSE = mean_squared_error(cur_rating, rating_predict)
        r_score = r2_score(cur_rating, rating_predict)

        # coef_full = coef_full[:, 0:-1]
        num_features = coef_back_in_rawspace.shape[1]  # number of data entries
        x = np.arange(num_features)  # the x locations for the groups

        # Prepare data
        y_mean = np.mean(coef_back_in_rawspace, axis=0)
        y_std = np.mean(coef_back_in_rawspace, axis=0)
        width = 0.5  # bar width

        # Plot
        fig, ax = plt.subplots()
        ax.bar(x, y_mean, width,  color='MediumSlateBlue', yerr=y_std,  error_kw={'ecolor': 'Tomato', 'linewidth': 2})
        axes = plt.gca()
        axes.set_ylim([-0.5, 0.5])
        ax.set_xticks(x+width)
        # ax.set_xticklabels(('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'))
        plt.show()
        string = 'dataset'+dataset_list[cur_dataset_ind]+'.jpg'
        plt.savefig(string)

    return


if __name__ == '__main__':
    main()
