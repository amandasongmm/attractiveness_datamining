import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bokeh.charts import Bar, output_file, show
import pandas as pd
import matplotlib.pyplot as matplt
import scipy.stats as spstat
import bokeh.plotting as plt
import numpy as np
import sklearn.manifold as manifold
import ujson

__author__ = 'amanda'


def load_feature_data(file_path):
    npz_data = np.load(file_path)
    feature_arr = npz_data['feature_arr']
    return feature_arr


def do_pca(feature_arr, explained_var=0.99):
    # preprocess the data
    fea_scaled = preprocessing.scale(feature_arr)

    # Compute PCA features
    pca = PCA(n_components=explained_var)
    pca_feature_arr = pca.fit_transform(fea_scaled)
    sorted_ind = np.argsort(pca_feature_arr, axis=1)
    return pca_feature_arr, sorted_ind


def face2d_scatter_bokeh_plot(locs, img_list, html_file_name):
    p = plt.figure(plot_width=2400, plot_height=1200)
    p.image_url(x=locs[:, 0], y=locs[:, 1], url=img_list, w=3, h=3, anchor="center")
    p.circle(x=locs[:, 0], y=locs[:, 1])
    plt.output_file(html_file_name)
    plt.save(p)
    plt.show(p)
    return


def matplot_1d_plot(imglist, indlist, nrows, ncols):
    fig, ax = plt.subplots(nrows=5, ncols=10)
    for curim in range(nrows*ncols):
        plt.subplot(nrows, ncols, curim)
        img = mpimg.imread()
    return


def load_imglist():
    with open('data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)

    return image_list


def my_main_4dataset_pipeline():
    whole_dataset_feature = load_feature_data('tmp/original_ConfiguralFeatures.npz')
    whole_img_list = load_imglist()
    dataset_name_list = ['MIT', 'gs', 'ngs', 'genhead']
    for cur_dataset_ind in range(4):
        input_feature = whole_dataset_feature[0+cur_dataset_ind*50: 50+cur_dataset_ind*50, :]
        pca_feature = do_pca(input_feature)
        locs = pca_feature[:, 0:2]
        img_list = whole_img_list[0+cur_dataset_ind*50:50+cur_dataset_ind*50]
        print img_list
        print locs.shape, len(img_list)
        save_file_name = './pc_visualization_fig/'+dataset_name_list[cur_dataset_ind]+'_pc_12.html'
    return


def my_4dataset_subplot():
    whole_dataset_feature = load_feature_data('tmp/original_ConfiguralFeatures.npz')
    whole_img_list = load_imglist()
    dataset_name_list = ['MIT', 'gs', 'ngs', 'genhead']
    for cur_dataset_ind in range(4):
        input_feature = whole_dataset_feature[0+cur_dataset_ind*50: 50+cur_dataset_ind*50, :]
        pca_feature, sorted_ind = do_pca(input_feature)
        img_list = whole_img_list[0+cur_dataset_ind*50 : 50+cur_dataset_ind*50]
        print img_list, len(img_list)
    return


my_4dataset_subplot()
