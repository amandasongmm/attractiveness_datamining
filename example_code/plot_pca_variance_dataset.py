import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bokeh.charts import Bar, output_file, show
import pandas as pd

'''
This program computes the PC components for each dataset's feature matrix.
It then analyze the explained variance captured by each PC in every separate dataset.
It generates two visualizations to illustrate the distribution of explained variance in each dataset.
'''

__author__ = 'amanda'


def main():
    # Specify a parameter
    explained_var = 0.95

    # Load data
    npz_data = np.load('tmp/full_rating.npz')
    feature_arr = npz_data['feature_arr']  # 200 * 29

    # Prepare a feature dataframe organized by: 'Explained_variance', 'PC Number', 'Dataset Origin'
    dataset_list = ['MIT', 'Glasgow', 'Mixed', 'genHead']
    fea_dataframe = pd.DataFrame(columns=['Explained_variance', 'PC Number', 'Dataset Origin'])

    # Iterate over 4 datasets, do a PCA analysis, report the explained variance of each PC in every dataset.
    for cur_dataset_ind in range(4):
        cur_fea = feature_arr[0+50*cur_dataset_ind: 50+cur_dataset_ind*50, :]
        cur_fea = preprocessing.scale(cur_fea)  # Preprocess the data, standardilzation.

        pca = PCA(n_components=explained_var)
        pca.fit(cur_fea)
        explained_ratio = pca.explained_variance_ratio_[:8]

        for cur_ind in range(explained_ratio.size):
            entry_variance = explained_ratio[cur_ind]
            pc_num = cur_ind + 1
            dataset_origin = dataset_list[cur_dataset_ind]
            fea_dataframe.loc[len(fea_dataframe)] = [entry_variance, pc_num, dataset_origin]

    # Compare four datasets' explained variance in the first 8 PCs
    p1 = Bar(fea_dataframe, label='PC Number', values='Explained_variance', title="Explained Variance by each PC",
             group='Dataset Origin', legend='top_right')
    output_file("./pc_variance_fig/all_dataset_pc_variance_plot1.html")
    show(p1)

    # Visualize the explained variance in each dataset.
    p2 = Bar(fea_dataframe, values='Explained_variance', title="Explained Variance by each PC", color="wheat")
    output_file("./pc_variance_fig/all_dataset_pc_variance_plot2.html")
    show(p2)
    return


if __name__ == '__main__':
    main()


