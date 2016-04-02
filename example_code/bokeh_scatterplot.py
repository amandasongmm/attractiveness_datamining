import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import bokeh.plotting as plt
import sklearn.manifold as manifold
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
    pca = PCA(n_components=explained_var, whiten=True)
    pca_feature_arr = pca.fit_transform(feature_arr)
    sorted_ind = np.argsort(pca_feature_arr, axis=0)
    return pca_feature_arr, sorted_ind


def embed_show(img_list, feats, html_file='dataset1_pc1.html'):
    # mapper = manifold.TSNE()
    # locs = mapper.fit_transform(feats)

    print feats.shape
    locs = np.asarray(feats)
    p = plt.figure(plot_width=2400, plot_height=1300)
    p.image_url(x=locs[:, 0]*2, y=locs[:, 0]*2, url=img_list, w=0.3, h=0.5, anchor="center")
    p.circle(x=locs[:, 0], y=locs[:, 1])

    plt.output_file(html_file, title="dataset, pc dim")
    plt.save(p)
    plt.show(p)
    return


def my_main(cur_ind):
    with open('data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)

    feature_list = load_feature_data('tmp/original_ConfiguralFeatures.npz')

    sub_fea = feature_list[0+cur_ind*50:50+cur_ind*50, :]
    pca_fea, sorted_ind = do_pca(sub_fea)
    embed_show(image_list[0+cur_ind*50:50+cur_ind*50], pca_fea)
    return pca_fea, sorted_ind

my_main(0)