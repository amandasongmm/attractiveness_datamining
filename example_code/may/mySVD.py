import numpy as np
import bokeh.plotting as plt
import sklearn.manifold as manifold
import pickle
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

    # reduce dimensions of Rater matrix and face matrix
    user = u[:, 0:num_dim]
    face = np.dot(s_diagonal[0:num_dim, 0:num_dim], v[0:num_dim, :])
    return user, face  # (1540, 30), (30, 200)


def tsne_on_user(u):
    model = manifold.TSNE()
    mapper = model.fit_transform(u)
    return mapper


def color_coding():
    with open(r"clusterDict.txt", "rb") as input_file:
        e = pickle.load(input_file)
    color_map_list = []
    for i in range(1540):
        if i in e[6]:
            cur_color = 'red'  # biggest cluster
        elif i in e[7]:
            cur_color = 'blue'  # second biggest cluster
        else:
            cur_color = 'green'  # others
        color_map_list.append(cur_color)

    return color_map_list


def scatter_user_show(mapper, colors):

    x = mapper[:, 0]
    y = mapper[:, 1]

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
    num_dim = 30
    user, face = svd_reconstruct(num_dim)

    # plot user scatter.
    mapper = tsne_on_user(user)
    colors = color_coding()
    scatter_user_show(mapper, colors)

    # u, s, v = np.linalg.svd(rating, full_matrices=False)
    #
    # with open('data/imagePathFile.json') as data_file:
    #     image_list = ujson.load(data_file)
    #
    # embed_show(image_list, v.T)

    return


if __name__ == '__main__':
    main()
