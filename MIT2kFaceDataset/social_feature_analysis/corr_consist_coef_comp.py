import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
from sklearn import cross_validation
from sklearn import linear_model
__author__ = 'amanda'

'''
Compute the consistency, correlation and coefficients to jointly understand how each feature jointly contribute to attractiveness prediction.
'''


def comp_consistency():
    # Load raw data
    file_name = '../Full Attribute Scores/psychology attributes/psychology-attributes.xlsx'
    df = pd.read_excel(open(file_name, 'rb'), sheetname=0)  # index starts from 0, which means the first sheet

    # Clean the data: delete irrelevant information, delete nan entries
    del_fields = ['Filename', 'catch', 'catchAns', 'subage', 'submale', 'subrace',
                         'catch.1', 'catchAns.1', 'subage.1', 'submale.1', 'subrace.1']
    df.drop(del_fields, inplace=True, axis=1)  # shortlist
    df.dropna(axis=0, how='any', inplace=True)  # delete any row that contains a nan

    # split raters into 2 groups, repeat 100 times.
    itr_num = 100
    img_num = df['Image #'].max()
    feat_names = list(df.columns.values)[1:]
    feat_num = len(feat_names)
    consist = np.zeros((feat_num, 1))

    for cur_itr in range(itr_num):
        print cur_itr
        ave_1 = np.zeros((img_num, feat_num))
        ave_2 = np.zeros((img_num, feat_num))

        for cur_im in range(1, img_num+1):
            tmp_data = df[df['Image #'] == cur_im]
            group_1, group_2 = train_test_split(tmp_data.values[:, 1:], test_size=0.5)
            ave_1[cur_im-1, :], ave_2[cur_im-1, :] = group_1.mean(axis=0), group_2.mean(axis=0)

        for cur_feat in range(feat_num):
            consist[cur_feat] += pearsonr(ave_1[:, cur_feat], ave_2[:, cur_feat])[0]

    consist /= np.float(itr_num)
    consist = [item for sublist in consist for item in sublist]  # flatten the list

    # save the data.
    df_consist = pd.DataFrame(consist, index=feat_names, columns=['consistency'])
    df_consist.to_csv('../clean_data/consistency.csv')
    return


def comp_correlation():
    full_soci_feats = pd.read_csv('../clean_data/reordered_social_feature.csv')
    attr = full_soci_feats['attractive'].values

    corr = []
    for cur_feat_ind in range(full_soci_feats.shape[1]):
        cur_feat = full_soci_feats.ix[:, cur_feat_ind].values
        corr.append(pearsonr(cur_feat, attr)[0])

    df_corr = pd.DataFrame(corr, index=feat_names, columns=['correlation'])
    df_corr.to_csv('../clean_data/correlation_with_attr.csv')


def comp_coeff():
    social_feats = pd.read_csv('../clean_data/reordered_social_feature.csv')
    attract_y = social_feats['attractive'].values
    feature_x = social_feats.drop(['attractive', 'unattractive'], axis=1)
    feat_names = list(feature_x.columns.values)
    feature_x = feature_x.values

    # run ridge regression
    itr_num = 50
    coef_list = np.zeros((len(feat_names), itr_num))

    for cur_itr in range(itr_num):
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(feature_x, attract_y, test_size=0.5)
        clf = linear_model.RidgeCV(alphas=np.logspace(-3, 1, num=20))
        clf.fit(x_train, y_train)

        coef_list[:, cur_itr] = clf.coef_

    coef_mean = coef_list.mean(axis=1)
    df_coef = pd.DataFrame(coef_mean, index=feat_names, columns=['coefficient'])
    df_coef.to_csv('../clean_data/coefficient_ave_social_fea.csv')


def compare_3c():
    df_consist = pd.read_csv('../clean_data/consistency.csv')
    df_corr = pd.read_csv('../clean_data/correlation_with_attr.csv')
    df_coef = pd.read_csv('../clean_data/coefficient_ave_social_fea.csv')

    

    return
