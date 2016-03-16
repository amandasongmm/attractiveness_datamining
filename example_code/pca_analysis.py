import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bokeh.charts import Bar, output_file, show
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.stats as spstat


__author__ = 'amanda'


def load_allfeatures_data():
    # Generate feature array from allFeatures.csv, if already generated, directly load orig_figure_arr
    if not os.path.exists('tmp/original_features.npz'):
        # Load feature array data, only keep the actual feature arrays
        orig_feature_arr = np.genfromtxt('data/allFeatures.csv', delimiter=',')[1:, :]
        orig_feature_arr = orig_feature_arr[:, 2:]
        np.savez('tmp/original_features', feature_arr=orig_feature_arr)  # Save the data for later direct loading.
    else:
        npz_data = np.load('tmp/original_features.npz')
        orig_feature_arr = npz_data['feature_arr']
    return orig_feature_arr


def calc_pca(orig_feature_arr, para_explained_var):
    if not os.path.exists('tmp/pca_data.npz'):
        # Preprocess the data, standardilzation.
        fea_scaled = preprocessing.scale(orig_feature_arr)  # Confirm if the scaled data has zero mean and unit variance

        # Compute PCA features
        pca = PCA(n_components=para_explained_var)
        pca_feature_arr = pca.fit_transform(fea_scaled)
        print pca_feature_arr.shape
        print '# of PCs needed to retain {} variance is {}.'.format(para_explained_var, pca_feature_arr.shape[1])

        # Plot the explained variance ratio by each PC.
        explained_ratio = pca.explained_variance_ratio_

        # Sort images according to ascending values in every PC dim
        pc_sorted_ind_list = np.argsort(pca_feature_arr, axis=0)  # from small to large, in each row.
        # Save the data.
        np.savez('tmp/pca_data', pca_feature_arr=pca_feature_arr, pc_sorted_ind_list=pc_sorted_ind_list,
                 explained_ratio=explained_ratio)

    else:
        npz_data = np.load('tmp/pca_data.npz')
        pca_feature_arr = npz_data['pca_feature_arr']
        pc_sorted_ind_list = npz_data['pc_sorted_ind_list']
        explained_ratio = npz_data['explained_ratio']
        print 'pc_sorted_ind_list shape is', pc_sorted_ind_list.shape
    return pca_feature_arr, pc_sorted_ind_list, explained_ratio


def plot_explained_variance(explained_ratio):
    fea_dataframe = pd.DataFrame(columns=['Explained_variance', 'PC Number'])
    for cur_ind in range(explained_ratio.size):
        entry_variance = explained_ratio[cur_ind]
        pc_num = cur_ind + 1
        fea_dataframe.loc[len(fea_dataframe)] = [entry_variance, pc_num]
    p1 = Bar(fea_dataframe, label='PC Number', values='Explained_variance', title="Explained Variance by each PC",
             xlabel="PC Number", ylabel="Explained Variance")
    output_file("./pc_variance_fig/new_feature_wholeset_pc_variance_bar.html")
    show(p1)
    return


def visualize_sorted_pc(pc_ind):
    subplot_num = 200
    plt.close('all')
    for pc_num in range(pc_ind.shape[1]):
        print 'curPC'+str(pc_num)+'\n'
        for cur_p in range(subplot_num):
            plt.subplot(10, 20, cur_p+1)
            cur_ind = pc_ind[cur_p, pc_num]
            img_link = './data/paddedImages/cropped'+str(cur_ind)+'.png'
            image = mpimg.imread(img_link)
            plt.axis('off')
            plt.imshow(image)
        save_name = 'pc_visualization_fig/paddedImage_pc'+str(pc_num)+'.png'
        plt.savefig(save_name, bbox_inches='tight')
    return


def correlation_comp(orig_feature, pca_feature):
    cor_arr = np.zeros((pca_feature.shape[1], orig_feature.shape[1]))
    pvalue_arr = np.zeros((pca_feature.shape[1], orig_feature.shape[1]))
    for cur_pca_ind in range(pca_feature.shape[1]):
        for cur_orig_ind in range(orig_feature.shape[1]):
            cor_coef, pvalue = spstat.pearsonr(pca_feature[:, cur_pca_ind], orig_feature[:, cur_orig_ind])
            cor_arr[cur_pca_ind, cur_orig_ind] = cor_coef
            pvalue_arr[cur_pca_ind, cur_orig_ind] = pvalue
    return cor_arr, pvalue_arr


def plot_correlation_heatmap(cor_arr, pvalue_arr):
    # read the column values of the orig feature arrays.
    orig_fea = pd.read_csv('data/allFeatures.csv', delimiter=',')
    title_list = list(orig_fea.columns.values)
    column_labels = title_list[2:]
    row_labels = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8']

    cor_arr[np.where(pvalue_arr > 0.05)] = 0
    cor_arr = cor_arr.T

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(cor_arr, cmap=plt.cm.bwr, alpha=0.8)
    ax.set_yticks(np.arange(cor_arr.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(cor_arr.shape[1])+0.5, minor=False)
    ax.xaxis.tick_top()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.xticks(rotation=45)
    plt.colorbar(heatmap)
    plt.savefig('./pc_variance_fig/pc_correlation_with_orig_feature.jpg', dpi=120, pad_inches=30)
    plt.show()
    return


def main(para_explained_var):
    orig_feature_arr = load_allfeatures_data()
    pca_feature_arr, pc_sorted_ind_list, explained_ratio = calc_pca(orig_feature_arr, para_explained_var)
    # plot_explained_variance(explained_ratio)
    # visualize_sorted_pc(pc_sorted_ind_list)
    cor_arr, pvalue_arr = correlation_comp(orig_feature_arr, pca_feature_arr)
    plot_correlation_heatmap(cor_arr, pvalue_arr)
    return


if __name__ == '__main__':
    explained_var = 0.95
    main(explained_var)
