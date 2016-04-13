import numpy as np


def load_data():
    # Load the rating data
    data = np.load('./tmp/pairwise_comparison_ratio.npz')
    i_over_j = data['i_over_j']
    j_over_i = data['j_over_i']
    i_equ_j = data['i_equ_j']
    return i_over_j, j_over_i, i_equ_j


def plot_bar(i, j):
    i_over_j, j_over_i, i_equ_j = load_data()

    return


