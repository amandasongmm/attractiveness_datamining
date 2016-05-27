import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from scipy.stats import spearmanr
from sklearn import preprocessing
import seaborn as sns
sns.set(context="paper", font="monospace")


soc_vals = pd.read_csv('../clean_data/reordered_social_feature.csv')
feats = pd.read_csv('../clean_data/geometric_features.csv')
feats.drop(['imgName'], axis=1, inplace=True)
feat_tag_list = list(feats.columns.values)
feats = preprocessing.scale(feats.values)

# Prepare parameters for ridge regression
itr_num = 50
alphas = np.logspace(-3, 2, num=20)
for cur_ind in range(soc_vals.shape[1]):
    cur_val = soc_vals.ix[:, cur_ind]
    cur_val_tag = cur_val.name

    data_list = list()
    for cur_itr in range(itr_num):
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(feats, cur_val.values, test_size=0.5)
        clf = linear_model.RidgeCV(alphas=alphas)
        clf.fit(x_train, y_train)
        y_test_pred = clf.predict(x_test)
        corr = spearmanr(y_test, y_test_pred)
        cur_data = {'coef': clf.coef_, 'intercept': clf.intercept_, 'alpha': clf.alpha_, 'corr': corr[0]}
        data_list.append(cur_data)

        coef_array = np.array([x['coef'] for x in data_list])

    print cur_val_tag, mean_list
