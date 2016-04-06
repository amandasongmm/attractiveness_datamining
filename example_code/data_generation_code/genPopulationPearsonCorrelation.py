import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
__author__ = 'amanda'


def load_data():
    # Load the rating data
    rating_data = np.load('../tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   # after transpose, the dimension is 200*1540
    return orig_rating  # 1540*200


def comp_pearson():
    orig_rating = load_data()
    rater_num = orig_rating.shape[0]
    corr = np.ones((rater_num, rater_num))
    pvalue = np.zeros((rater_num, rater_num))

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
    np.savez('../tmp/pearsonCorrelation', corr=corr, pvalue=pvalue)
    print 'Pearson Done.\n'
    return corr, pvalue


def comp_spearmanr():
    orig_rating = load_data()
    print orig_rating.shape
    corr, pvalue = ss.spearmanr(orig_rating.T)
    print corr.shape
    np.savez('../tmp/spearmanRankCorrelation', corr=corr, pvalue=pvalue)
    print 'Spearmanr rank correlation done.\n'
    return corr, pvalue


# comp_pearson()
comp_spearmanr()

