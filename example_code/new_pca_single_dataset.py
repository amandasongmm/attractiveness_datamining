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
    # # preprocess the data
    # feature_arr = preprocessing.scale(feature_arr)

    # Compute PCA features
    pca = PCA(n_components=explained_var)
    pca_feature_arr = pca.fit_transform(feature_arr)
    sorted_ind = np.argsort(pca_feature_arr, axis=1)
    return pca_feature_arr, sorted_ind


def embed_show(img_list, feats, html_file):
    # mapper = manifold.TSNE()
    # locs = mapper.fit_transform(feats)
    print feats.shape
    # locs = feats[:, 0:2]
    p = plt.figure(plot_width=2400, plot_height=1300)
    # p.image_url(x=locs[:, 0]*2, y=locs[:, 0]*0.5, url=img_list, w=0.2, h=0.3, anchor="center")
    p.image_url(x=feats*2, y=feats*0.5, url=img_list, w=0.2, h=0.3, anchor="center")
    # p.circle(x=locs[:, 0], y=locs[:, 1])
    p.circle(x=feats*2, y=feats*0.5)

    plt.output_file(html_file, title=html_file)
    plt.save(p)
    plt.show(p)
    return


def my_main(cur_dataset_ind, pc_dim):
    # calculate pca features
    feature_list = load_feature_data('tmp/original_ConfiguralFeatures.npz')
    sub_fea = feature_list[0+cur_dataset_ind*50:50+cur_dataset_ind*50, :]
    pca_fea, sorted_ind = do_pca(sub_fea)
    cur_feature = pca_fea[:, pc_dim-1]

    # select img_list
    with open('data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)
    cur_img_list = image_list[0+cur_dataset_ind*50:50+cur_dataset_ind*50]

    # filename
    dataset_name_list = ['MIT', 'gs', 'ngs', 'genhead']
    save_file_name = './single_dataset_pc_sanity_check/'+dataset_name_list[cur_dataset_ind]+'_pc_'+str(pc_dim)+'.html'

    # plot
    embed_show(cur_img_list, cur_feature, save_file_name)
    return


my_main(0, 1)