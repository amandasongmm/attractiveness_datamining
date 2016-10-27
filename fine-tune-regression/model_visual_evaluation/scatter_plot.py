import numpy as np
import matplotlib.pyplot as plt

__author__ = 'amanda'


def my_scatter_plot(result_path, fig_title, fig_save_path):

    pred_result = np.load(result_path)
    human_rating = pred_result['label']
    model_rating = pred_result['outputs']
    model_rating = model_rating.reshape((len(model_rating),))  # (n, 1) to (n, )

    cor = np.corrcoef(human_rating, model_rating)[0, 1]

    plt.plot(human_rating, model_rating, 'b.')
    plt.xlabel('Human rating')
    plt.ylabel('Model prediction')
    plt.grid(True)
    plt.xlim(2, 9)
    plt.ylim(2, 9)
    plt.gca().set_aspect('equal', adjustable='box')
    corr_number = ':correlation {0:.2f}'.format(cor)
    plt.title(fig_title+corr_number)
    plt.savefig(fig_save_path)

    return

my_scatter_plot('train1.npz', 'Training data', 'train_scatter.png')
my_scatter_plot('test1.npz', 'test data', 'test_scatter.png')
