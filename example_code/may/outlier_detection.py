import numpy as np
import pandas as pd
import bokeh.plotting as plt
import pickle

__author__ = 'amanda'


def load_group_meta_data():

    df = pd.read_csv('./intermediate_metadata/selfcorrelation_pearsonVals.csv')
    pears = df['pears']
    self_correlation = pears.as_matrix()

    tmp = np.load('./intermediate_metadata/group_correlation.npz')
    group_correlation = tmp['group_corr_array']

    tmp = np.load('./intermediate_metadata/rater_info.npz')
    sex_ind = tmp['sex_ind']
    age_ind = tmp['age_ind']
    twin_ind = tmp['twin_ind']

    return sex_ind, age_ind, twin_ind, group_correlation, self_correlation


def color_coding():
    with open(r"clusterDict.txt", "rb") as input_file:
        e = pickle.load(input_file)
    color_map_list = []
    for i in range(1540):
        if i in e[6]:
            cur_color = 'red'  # biggest cluster
        elif i in e[7]:
            cur_color = 'blue'  # second biggest cluster
        else:
            cur_color = 'green'  # others
        color_map_list.append(cur_color)
    color_map_by_cluster = color_map_list
    return color_map_by_cluster


def scatter_plot(x, y, xlabel, ylabel, title):

    p = plt.figure()
    p.scatter(x, y, fill_color=colors, fill_alpha=0.6, line_color=None)
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    plt.output_file(title, title=title)
    plt.show(p)
    return


def individual_check(check_list):
    sum(i > 0.65 for i in check_list)
    return


def my_main_1():
    # main
    colors = color_coding()
    sex_ind, age_ind, twin_ind, group_correlation, self_correlation = load_group_meta_data()

    # plot 1
    x = group_correlation
    y = self_correlation
    xlabel = 'group_correlation'
    ylabel = 'self correlation'
    title = xlabel + 'vs' + ylabel + '.html'
    scatter_plot(x, y, xlabel, ylabel, title)
    return


def my_main2():
    sex_ind, age_ind, twin_ind, group_correlation, self_correlation = load_group_meta_data()
    rating_data = np.load('./tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']

    female_group = orig_rating[sex_ind == 1, :]  # 1 for female
    male_group = orig_rating[sex_ind == 2, :]

    female_ave = np.mean(female_group, axis=0)
    female_average = female_ave.tolist()

    male_ave = np.mean(male_group, axis=0)
    male_average = male_ave.tolist()

    def group_correlation_comp(group_data, ave_data):
        group_corr_array = []
        for i in range(group_data.shape[0]):
            cov = np.corrcoef(ave_data, orig_rating[i, :])
            group_corr_array.append(cov[0, 1])
        return group_corr_array



    return






