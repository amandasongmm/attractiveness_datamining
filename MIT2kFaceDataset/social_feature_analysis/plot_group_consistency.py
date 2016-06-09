import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

__author__ = 'amanda'


twin_ratings = np.load('../../example_code/tmp/clean_rating_data.npz')
twin_ratings = twin_ratings['full_rating']

group_1, group_2 = train_test_split(twin_ratings, test_size=0.5)
rho, pval = spearmanr(group_1.mean(axis=0), group_2.mean(axis=0))

plt.scatter(group_1.mean(axis=0), group_2.mean(axis=0), alpha=0.5)
title_str = '{}{:.2f}'.format('Correlation between 2 group''s average opinion=', rho)
plt.xlabel('Group 1: mean ratings for 200 faces')
plt.ylabel('Group 2: mean ratings for 200 faces')
plt.title(title_str)
plt.savefig('../figs/group_consistency')
plt.show()

