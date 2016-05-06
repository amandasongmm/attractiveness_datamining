import numpy as np
import sklearn.manifold as manifold
from sklearn.cluster import KMeans
from bokeh.plotting import figure, show, output_file, save
import bokeh.plotting as plt
import time
# import matplotlib.pyplot as plt
import ujson
__author__ = 'amanda'


def prepare_data(u_keep_dim_num):
    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   #

    # Conduct svd
    u, s, v = np.linalg.svd(orig_rating, full_matrices=False)

    # Conduct tsne
    model = manifold.TSNE()
    output = model.fit_transform(u[:, 0:u_keep_dim_num])
    return output


def cluster_user(input_data, num_cluster):
    x_label = KMeans(n_clusters=num_cluster).fit_predict(input_data)
    return x_label


def embed_show(label, feat):
    colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'gray'}
    colors = [colormap[x] for x in label]

    p = figure(title='Visualize clusters')
    p.circle(x=feat[:, 0], y=feat[:, 1], color=colors, fill_alpha=0.2, size=10)
    output_file('user_cluster.html', title='user cluster')
    show(p)
    return


def embed_show_main():
    u_keep_dim = 2
    kmeans_cluster_num = 4
    tsne_return_data = prepare_data(u_keep_dim)
    labels = cluster_user(tsne_return_data, num_cluster=kmeans_cluster_num)

    sort_ind = np.argsort(labels)
    # embed_show(labels, tsne_return_data)
    return


def correlation_map_remapping():
    # load corr and pvalue
    file_path = 'tmp/pearsonCorrelation.npz'
    tmp_data = np.load(file_path)
    corr = tmp_data['corr']   # 1540 * 1540
    pvalue = tmp_data['pvalue']

    corr = corr[:, index]
    corr = corr[index, :]
    pvalue = pvalue[:, index]
    pvalue = pvalue[index, :]

    corr[np.where(pvalue > 0.05)] = 0
    a = 12
    print 'Data preparaion done. Now start ploting...\n'
    start = time.time()
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(corr, cmap=plt.cm.bwr, alpha=0.8, vmin=-1, vmax=1)
    plt.colorbar(heatmap)
    plt.title('New Pearson Correlation Heatmap')
    plt.savefig('./correlationMap/'+'new Pearson Correaltion map')
    end = time.time()
    print 'time elapsed = '+str(end-start) + ' sec'
    return


def reorganize_rating():
    # prepare index.
    u_keep_dim = 2
    kmeans_cluster_num = 4
    tsne_return_data = prepare_data(u_keep_dim)
    labels = cluster_user(tsne_return_data, num_cluster=kmeans_cluster_num)
    index = np.argsort(labels)

    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']

    t = labels[index]
    group_mean = []
    for i in range(kmeans_cluster_num):
        cur_ind = np.where(t == i)
        tmp = orig_rating[cur_ind, :]
        tmp = np.reshape(tmp, [tmp.shape[1], tmp.shape[2]])
        tmp = np.mean(tmp, axis=0)
        group_mean.append(tmp)
    return group_mean


def plot_group_attractiveness():
    start = time.time()
    group_mean = reorganize_rating()
    end = time.time()
    a = 12
    with open('data/imagePathFile.json') as data_file:
        img_list = ujson.load(data_file)

    # locs = np.asarray(feats)*40
    p = plt.figure()
    p.image_url(x=group_mean[2], y=group_mean[3], url=img_list, w=0.2, h=0.3, anchor="center")

    p.circle(x=group_mean[2], y=group_mean[3])

    plt.output_file('cluster1vs2.html', title="cluster1 vs 2")
    plt.save(p)
    plt.show(p)
    return


plot_group_attractiveness()







