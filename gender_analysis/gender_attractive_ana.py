import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from scipy.stats import spearmanr
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set(context="paper", font="monospace")


def gen_gender_split_data():

    def select_gender(data, gender_ind):
        gen_list = np.where(gender_list == gender_ind)
        gen_list = list(itertools.chain.from_iterable(gen_list))
        gen_data = data.iloc[gen_list, :]
        return gen_list, gen_data

    # load social feature array (2222 * 40)
    social_fea = pd.read_pickle('../MIT2kFaceDataset/clean_data/feature_array')

    # load gender info
    file_name = '../MIT2kFaceDataset/Full Attribute Scores/demographic & others labels/demographic-others-labels.xlsx'
    xl_file = pd.ExcelFile(file_name)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    gender_list = dfs['Final Values']['Gender']

    fe_list, fe_social_fea = select_gender(social_fea, 0)
    ma_list, ma_social_fea = select_gender(social_fea, 1)

    np.savez('gender_list', fe_list=fe_list, ma_list=ma_list)
    pd.to_pickle(fe_social_fea, 'fe_social_feature')
    pd.to_pickle(ma_social_fea, 'ma_social_feature')
    return fe_social_fea, ma_social_fea


def load_data():
    all_fea = pd.read_pickle('../MIT2kFaceDataset/clean_data/feature_array')
    fe_social_fea = pd.read_pickle('fe_social_feature')
    ma_social_fea = pd.read_pickle('ma_social_feature')
    return fe_social_fea, ma_social_fea, all_fea


def lr(data):
    attract_y = data['attractive'].values

    feature_x = data.drop(['attractive', 'unattractive'], axis=1)
    # x_fields = list(feature_x.columns.values)
    feature_x = feature_x.values

    corr_array = []
    itr_num = 100
    for itr in range(itr_num):
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(feature_x, attract_y, test_size=0.5)
        clf = linear_model.RidgeCV(alphas=np.logspace(-4, 10, num=30), fit_intercept=True)
        clf.fit(x_train, y_train)

        y_test_predict = clf.predict(x_test)
        corr = spearmanr(y_test, y_test_predict)
        corr_array.append(corr[0])

    ave_correlation = np.mean(corr_array)
    return ave_correlation


def correlation_heat_map(data, save_name):
    corr_mat = data.corr()

    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_mat, square=True, vmin=-1, vmax=1)

    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.tick_params(labelsize=10)
    plt.savefig(save_name+'.png')
    plt.show()
    return


def correlation_cmp():
    fe_data, ma_data, all_data = load_data()
    df = pd.concat([fe_data.corr()['attractive'], ma_data.corr()['attractive'], all_data.corr()['attractive']],
                   join='outer', axis=1)
    df.columns = ['fe', 'ma', 'all']
    pd.to_pickle(df, 'correlation_df_for_allgen_separate_gen')
    return df


def consistency_cmp():

    def single_gen(gen_list):
        consist = np.zeros((feat_num, 1))

        for cur_itr in range(itr_num):
            print 'cur itr={}'.format(cur_itr)
            ave_1 = np.zeros((len(gen_list), feat_num))
            ave_2 = np.zeros((len(gen_list), feat_num))

            for count_ind, cur_im in enumerate(gen_list):
                im_ind = cur_im + 1
                tmp_data = df[df['Image #'] == im_ind]
                group1, group2 = cross_validation.train_test_split(tmp_data.values[:, 1:], test_size=0.5)
                ave_1[count_ind, :], ave_2[count_ind, :] = group1.mean(axis=0), group2.mean(axis=0)

            for cur_feat in range(feat_num):
                consist[cur_feat] += spearmanr(ave_1[:, cur_feat], ave_2[:, cur_feat])[0]

        consist = [item/np.float(itr_num) for sublist in consist for item in sublist]  # flatten the list
        return consist

    # load gender list.
    gender_list = np.load('gender_list.npz')
    fe_list = gender_list['fe_list']
    ma_list = gender_list['ma_list']

    # Load raw rating data.
    file_name = '../MIT2kFaceDataset/Full Attribute Scores/psychology attributes/psychology-attributes.xlsx'
    df = pd.read_excel(open(file_name, 'rb'), sheetname=0)  # index starts from 0, which means the first sheet

    # Clean the data: delete irrelevant information, delete nan entries
    del_fields = ['Filename', 'catch', 'catchAns', 'subage', 'submale', 'subrace', 'catch.1', 'catchAns.1',
                  'subage.1', 'submale.1', 'subrace.1']
    df.drop(del_fields, inplace=True, axis=1)  # shortlist
    df.dropna(axis=0, how='any', inplace=True)  # delete any row that contains a nan

    # split raters into 2 groups, repeat 100 times.
    itr_num = 100
    feat_names = list(df.columns.values)[1:]
    feat_num = len(feat_names)

    # compute consistency for female and male faces.
    fe_face_consist = single_gen(fe_list)
    ma_face_consist = single_gen(ma_list)

    joint_consist = [fe_face_consist, ma_face_consist]
    joint_consist = np.asarray(joint_consist)
    joint_consist = np.transpose(joint_consist)
    joint_consist = pd.DataFrame(joint_consist, index=feat_names, columns=['female', 'male'])

    # save the consistency data.
    pd.to_pickle(joint_consist, 'consistency')
    return


def plot_consistency():
    consist = pd.read_pickle('consistency')

    return


def main():
    fe_data, ma_data, all_data = load_data()
    fe_score = lr(fe_data)
    ma_score = lr(ma_data)
    print 'Female score = {}, male score = {}'.format(fe_score, ma_score)
    return


if __name__ == "__main__":
    start_t = time.time()
    consistency_cmp()
    print 'Elapsed time ={}'.format(time.time()-start_t)




