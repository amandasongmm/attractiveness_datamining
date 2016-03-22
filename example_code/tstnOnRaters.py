import numpy as np
from sklearn.manifold import TSNE
from bokeh.charts import Scatter, output_file, show
import pandas as pd
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt

__author__ = 'amanda'


def load_data():
    # Load the rating data
    ds = np.load('tmp/coefficient.npz')
    coef_all = ds['coef']
    print coef_all.shape  # 1540*9
    return coef_all
    # print orig_rating.shape
    # return orig_rating  # the dimension is 1540*200


def do_tsne(x):
    model = TSNE(n_components=2, random_state=0)
    tsn_feature = model.fit_transform(x)
    np.savez('tmp/tsneFeature_input_lr_coefficient', tsn_feature=tsn_feature)
    return tsn_feature


def scatterplot(tsne_features):
    df = pd.DataFrame(tsne_features, columns=['tsne1', 'tsne2'])
    p = Scatter(df, x='tsne1', y='tsne2', marker='square', title='Linear Regression Coefficient visualization',
                legend='top_left', xlabel='tsne1', ylabel='tsne2')
    output_file("Linear Regression Coefficient clusters.html")
    show(p)
    return


def distance_comp(tsne_features):  # 1540*2
    rater_num = tsne_features.shape[0]
    a1 = tsne_features[0:rater_num/2, :]
    a2 = tsne_features[rater_num/2:, :]
    distance_matrix = sd.cdist(a1, a2)
    plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
    plt.show()
    return


def main():
    coef = load_data()
    rater_array_new = do_tsne(coef)  # (n_sample, 2)
    scatterplot(rater_array_new)
    distance_comp(rater_array_new)


if __name__ == '__main__':
    main()
