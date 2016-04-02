import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
__author__ = 'amanda'


def load_data():
    # Load the rating data
    rating_data = np.load('tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   # after transpose, the dimension is 200*1540
    return orig_rating  # 1540*200


def comp_pearson():
    orig_rating = load_data()
    rater_num = orig_rating.shape[0]
    corr = np.ones((rater_num, rater_num))
    pvalue = np.ones((rater_num, rater_num))

# compute the upper triangle
    for rater_1 in range(rater_num):
        print 'Now computing row{}...\n'.format(rater_1+1)
        for rater_2 in range(rater_1+1, rater_num):
            rating_1 = orig_rating[rater_1, :]
            rating_2 = orig_rating[rater_2, :]
            corr[rater_1, rater_2], pvalue[rater_1, rater_2] = ss.pearsonr(rating_1, rating_2)

# fill in the whole matrix
    irows, icols = np.triu_indices(len(corr), 1)
    corr[icols, irows] = corr[irows, icols]
    pvalue[icols, irows] = pvalue[irows, icols]
    return corr, pvalue


def plot_correlation_heatmap():
    cor_arr, pvalue_arr = comp_pearson()
    cor_arr[np.where(pvalue_arr > 0.05)] = 0

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(cor_arr, cmap=plt.cm.bwr, alpha=0.8)
    ax.set_yticks(np.arange(cor_arr.shape[0]), minor=False)
    ax.set_xticks(np.arange(cor_arr.shape[1]), minor=False)
    plt.colorbar(heatmap)
    plt.show()
    plt.savefig('test.jpg', dpi=120, pad_inches=30)
    return

plot_correlation_heatmap()


# def cal_population_rank_correlation():  # with p value return, but that's too slow.
#     orig_rating = load_data()
#     rater_num = orig_rating.shape[0]
#
#     # Kendall's Tau
#     kendall_tau = np.zeros((rater_num, rater_num))
#     kendall_tau_pvalue = np.zeros((rater_num, rater_num))
#     for rater_1 in range(rater_num):
#         for rater_2 in range(rater_1+1, rater_num):
#             rating_1 = orig_rating[rater_1, :]
#             rating_2 = orig_rating[rater_2, :]
#             kendall_tau[rater_1, rater_2], kendall_tau_pvalue[rater_1, rater_2] = ss.kendalltau(rating_1, rating_2)
#         print 'row {} done...\n'.format(rater_1)
#     return kendall_tau, kendall_tau_pvalue

# Try to use pandas's default function to comput kendall's tau, but the speed is also very very slow.
# def cal_correlation(method):
#     print 'Now computing {} correlation...\n'.format(method)
#     orig_rating = load_data()
#     frame = pd.DataFrame(orig_rating)
#     correlation_matrix = frame.corr(method=method)
#     return correlation_matrix

# pearson_cor = cal_correlation('pearson')
# # kendall_cor = cal_correlation('kendall')
# # spearman_cor = cal_correlation('spearman')
# np.savez('../tmp/rater_correlation', pearson_cor=pearson_cor, kendall_cor=kendall_cor, spearman_cor=spearman_cor)


