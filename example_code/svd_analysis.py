import numpy as np
import bokeh.plotting as plt
import sklearn.manifold as manifold
import ujson
import time
from sklearn.metrics import mean_squared_error
np.seterr(divide='ignore', invalid='ignore')

__author__ = 'amanda'


def load_data():
    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   #
    return orig_rating  # 1540 * 200


def svd_reconstruct(num_dim):
    rating = load_data()
    u, s, v = np.linalg.svd(rating, full_matrices=False)     # U.shape = [1540*200],s.shape = (200,) V.shape = 200*200
    s_diagonal = np.diag(s)

    # Use a subset of singular values to reconstruct the matrix.
    reconstructed_rating = np.dot(u[:, 0:num_dim], np.dot(s_diagonal[0:num_dim, 0:num_dim], v[0:num_dim, :]))

    # Compute the reconstruction error RMSE
    recon_err = mean_squared_error(rating, reconstructed_rating)
    print 'dim = {}, recon_err ={:.2f}'.format(num_dim, recon_err)
    return recon_err


def cluster_user(u):
    model = manifold.TSNE()
    mapper = model.fit_transform(u)
    # mapper = u
    x = mapper[:, 0]
    y = mapper[:, 1]
    colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]
    tools = "resize,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,previewsave,box_select," \
            "poly_select,lasso_select"
    p = plt.figure(tools=tools)
    p.scatter(x, y, fill_color=colors, fill_alpha=0.6, line_color=None)
    plt.output_file('user_scatter.html', title='user scatter plot example')
    plt.show(p)
    return


def embed_show(img_list, feats):
    mapper = manifold.TSNE()
    locs = mapper.fit_transform(feats)

    print feats.shape
    # locs = np.asarray(feats)*40
    p = plt.figure()
    p.image_url(x=locs[:, 0], y=locs[:, 0], url=img_list, w=5, h=8, anchor="center")

    p.circle(x=locs[:, 0], y=locs[:, 1])

    plt.output_file('test.html', title="dataset, pc dim")
    plt.save(p)
    plt.show(p)
    return


def main():
    num_dim = 50
    svd_reconstruct(num_dim)

    # u, s, v = np.linalg.svd(rating, full_matrices=False)
    #
    # with open('data/imagePathFile.json') as data_file:
    #     image_list = ujson.load(data_file)
    #
    # embed_show(image_list, v.T)

    return


if __name__ == '__main__':
    main()
