import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bokeh.charts import Bar, output_file, show
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as spstat


__author__ = 'amanda'


def load_orig_feature_data(file_path):
    npz_data = np.load(file_path)
    orig_feature_arr = npz_data['feature_arr']
    return orig_feature_arr


def calc_pca(orig_feature_arr, para_explained_var):
    # preprocess the data
    fea_scaled = preprocessing.scale(orig_feature_arr)

    # Compute PCA features
    pca = PCA(n_components=para_explained_var)
    pca_feature_arr = pca.fit_transform(fea_scaled)
    print pca_feature_arr.shape, '# of PCs needed to retain {} variance is {}.'\
        .format(para_explained_var, pca_feature_arr.shape[1])

    # Plot the explained variance ratio by each PC.
    explained_ratio = pca.explained_variance_ratio_

    # # Save the data.
    # np.savez('tmp/pca_data', pca_feature_arr=pca_feature_arr, explained_ratio=explained_ratio)
    return pca_feature_arr, explained_ratio


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
    orig_fea = pd.read_csv('data/allConfiguralFeatures.csv', delimiter=',')
    title_list = list(orig_fea.columns.values)
    column_labels = title_list[2:]
    pc_num = cor_arr.shape[0]
    print pc_num
    row_labels = ['pc'+str(i+1) for i in range(pc_num)]

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
    plt.show()
    plt.savefig('./pc_variance_fig/pc_correlation_with_orig_feature.jpg', dpi=120, pad_inches=30)

    return


def main(para_explained_var, filepath):
    orig_feature_arr = load_orig_feature_data(filepath)
    pca_feature_arr, explained_ratio = calc_pca(orig_feature_arr, para_explained_var)
    plot_explained_variance(explained_ratio)
    cor_arr, pvalue_arr = correlation_comp(orig_feature_arr, pca_feature_arr)
    plot_correlation_heatmap(cor_arr, pvalue_arr)
    return


if __name__ == '__main__':
    explained_var = 0.99
    orig_feature_file_path = 'tmp/original_ConfiguralFeatures.npz'
    main(explained_var, orig_feature_file_path)
