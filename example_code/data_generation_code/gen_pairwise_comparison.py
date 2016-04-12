import numpy as np

__author__ = 'amanda'


def load_rating_data():
    rating_data = np.load('../tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']
    return orig_rating  # 1540*200


def pair_comp():
    orig_rating = load_rating_data()
    img_num = orig_rating.shape[1]

    i_over_j = np.zeros((img_num, img_num))
    j_over_i = np.zeros((img_num, img_num))
    i_equ_j = np.zeros((img_num, img_num))
    for i in range(img_num):
        print 'cur index=', i
        for j in range(img_num):
            cur_i = orig_rating[:, i]
            cur_j = orig_rating[:, j]
            i_over_j[i, j] = sum(cur_i > cur_j)
            j_over_i[i, j] = sum(cur_i < cur_j)
            i_equ_j[i, j] = sum(cur_i == cur_j)

    np.savez('../tmp/pairwise_comparison_ratio', i_over_j=i_over_j, j_over_i=j_over_i, i_equ_j=i_equ_j)
    return i_over_j, j_over_i, i_equ_j

pair_comp()
