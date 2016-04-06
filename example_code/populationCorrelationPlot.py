import numpy as np
import matplotlib.pyplot as plt
import time
__author__ = 'amanda'


def load_data(correlation_key):
    if correlation_key == 'pearson':
        filepath = 'tmp/pearsonCorrelation.npz'
    elif correlation_key == 'spearman':
        filepath = 'tmp/spearmanRankCorrelation.npz'
    else:
        print 'check the data path or your key name.'
    tmp_data = np.load(filepath)
    corr = tmp_data['corr']   # 1540 * 1540
    pvalue = tmp_data['pvalue']
    return corr, pvalue  # 1540*1540


def plot_correlation_heatmap(key):
    cor_arr, pvalue_arr = load_data(key)
    cor_arr[np.where(pvalue_arr > 0.05)] = 0
    print 'Data loading done. Now start ploting...\n'
    start = time.time()
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(cor_arr, cmap=plt.cm.bwr, alpha=0.8, vmin=-1, vmax=1)
    plt.colorbar(heatmap)
    plt.title(key+' Correlation Heatmap')
    plt.savefig('./correlationMap/'+key)
    end = time.time()
    print 'time elapsed = '+str(end-start) + ' sec'
    return


def cal_average_corr(key):
    corr, pvalue = load_data(key)
    corr_mean = np.mean(corr[np.where(pvalue < 0.05)])
    print 'The {} correlation is: {:.2f}'.format(key, corr_mean)

# plot_correlation_heatmap('pearson')
# plot_correlation_heatmap('spearman')
cal_average_corr('pearson')
cal_average_corr('spearman')





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


