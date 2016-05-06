import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_validation import train_test_split
import bokeh.plotting as plt
import ujson
from scipy.stats import pearsonr
np.seterr(divide='ignore', invalid='ignore')
__author__ = 'amanda'


def load_data():
    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   # after transpose, the dimension is 1540 * 200.

    # Load the configural feature data:
    npz_data = np.load('tmp/original_ConfiguralFeatures.npz')  # Chicago feature
    configural_feature_arr = npz_data['feature_arr']

    # Load the pixel feature data.
    npz_data = np.load('./tmp/pixel_features.npz')
    pixel_feature_arr = npz_data['pixel_feature_arr'].T
    return orig_rating, configural_feature_arr,  pixel_feature_arr  # 1540*200, 200 * 29, 200 * 4000


def do_pca(feature_arr, explained_var=0.95):
    # preprocess the data
    feature_arr = preprocessing.scale(feature_arr)

    # Compute PCA features
    pca = PCA(n_components=explained_var, whiten=True)
    pca_feature_arr = pca.fit_transform(feature_arr)

    return pca_feature_arr


def do_lr(feature_arr, rating):
    x_train, x_test, y_train, y_test = train_test_split(feature_arr, rating, test_size=0.1, random_state=42)

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    # Model prediction
    # y_train_predict = regr.predict(x_train)
    # y_test_predict = regr.predict(x_test)
    y_full_predict = regr.predict(feature_arr)
    return y_full_predict


def do_scatter_plot(ave_rating, pred_rating, img_list, html_file, corr):
    p = plt.figure(title=str(corr), x_range=(1, 7), y_range=(1, 7), x_axis_label='Actual Rating',
                   y_axis_label='Predicted Rating')
    p.image_url(x=ave_rating, y=pred_rating, url=img_list, w=0.2, h=0.3, anchor='center')
    p.circle(x=ave_rating, y=pred_rating)

    plt.output_file(html_file, title=html_file)
    plt.save(p)
    plt.show(p)
    return


def iterate_over_datasets(feature_arr, average_rating, fea_name, image_list):
    print 'now computing ' + fea_name
    for cur_dataset_ind in range(4):
        sub_fea = feature_arr[0+cur_dataset_ind*50:50+cur_dataset_ind*50, :]
        pca_fea = do_pca(sub_fea)

        y_rating = average_rating[0+cur_dataset_ind*50:50+cur_dataset_ind*50]

        y_predict = do_lr(pca_fea, y_rating)
        corr, p = pearsonr(y_predict, y_rating)

        sub_img_list = image_list[0+cur_dataset_ind*50:50+cur_dataset_ind*50]

        save_name = fea_name+'dataset'+str(cur_dataset_ind+1)+'.html'
        do_scatter_plot(y_rating, y_predict, sub_img_list, save_name, corr)
    return


def my_main1():
    with open('data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)

    rating, configural_feature,  pixel_feature = load_data()
    average_rating = np.mean(rating, axis=0)

    # fea_name = 'configural'
    # iterate_over_datasets(configural_feature, average_rating, fea_name, image_list)

    fea_name = 'pixel'
    iterate_over_datasets(pixel_feature, average_rating, fea_name, image_list)
    return


my_main1()
