import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import bokeh.plotting as plt
import ujson


__author__ = 'amanda'


def load_feature_data(file_path):
    npz_data = np.load(file_path)
    feature_arr = npz_data['feature_arr']
    return feature_arr


def do_pca(feature_arr, explained_var=0.99):
    # preprocess the data
    feature_arr = preprocessing.scale(feature_arr)

    # Compute PCA features
    # pca = PCA(n_components=explained_var, whiten=True)
    pca = PCA(n_components=explained_var, whiten=False)
    pca_feature_arr = pca.fit_transform(feature_arr)
    sorted_ind = np.argsort(pca_feature_arr, axis=0)
    return pca_feature_arr, sorted_ind


def embed_show(img_list, feats, html_file, feature_dim):

    locs = np.asarray(feats)
    p = plt.figure(plot_width=2400, plot_height=1300)
    scale = 5
    p.image_url(x=locs[:, feature_dim]*scale, y=locs[:, feature_dim]*scale, url=img_list, w=1, h=1.5, anchor="center")
    p.circle(x=locs[:, feature_dim]*scale, y=locs[:, feature_dim]*scale)

    plt.output_file(html_file, title=html_file)
    plt.save(p)
    plt.show(p)
    return


def my_main(cur_pc):
    with open('data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)

    feature_list = load_feature_data('tmp/original_ConfiguralFeatures.npz')
    for cur_dataset_ind in range(1):
        sub_fea = feature_list[0+cur_dataset_ind*50:50+cur_dataset_ind*50, :]
        pca_fea, sorted_ind = do_pca(sub_fea)

        sub_img_list = image_list[0+cur_dataset_ind*50:50+cur_dataset_ind*50]

        save_name = 'dataset'+str(cur_dataset_ind+1)+'pc'+str(cur_pc)+'.html'
        embed_show(sub_img_list, pca_fea, save_name, cur_pc-1)
    return pca_fea, sorted_ind


for pc_num in range(1):
    my_main(pc_num)
