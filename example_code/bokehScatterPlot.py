import bokeh.plotting as plt
import numpy as np
import sklearn.manifold as manifold
import ujson

__author__ = 'amanda'


def load_data():
    data = np.load('tmp/pca_data.npz')
    pca_feature = data['pca_feature_arr']
    return pca_feature


def embed_show(img_list, feats, html_file_name='figure.html'):
    # mapper = manifold.TSNE()
    # locs = mapper.fit_transform(feats)
    locs = feats
    p = plt.figure(plot_width=2400, plot_height=1200)
    p.image_url(x=locs[:, 0], y=locs[:, 1], url=img_list, w=0.5, h=0.5, anchor="center")
    p.circle(x=locs[:, 0], y=locs[:, 1])

    plt.output_file(html_file_name)
    plt.save(p)
    plt.show(p)
    return


if __name__ == '__main__':
    with open('data/imagePathFile.json') as data_file:
        image_list = ujson.load(data_file)
    feature_list = load_data()

    save_file_name = './pc_visualization_fig/pc_12.html'
    loc = feature_list[:, 0:2]  # visualize only the first two dimensions.
    embed_show(image_list, feature_list)
