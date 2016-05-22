import pandas as pd
import simplejson
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

__author__ = 'amanda'

file_name = '../Full Attribute Scores/psychology attributes/psychology-attributes.xlsx'
df = pd.read_excel(open(file_name, 'rb'), sheetname=0)  # index starts from 0, which means the first sheet


''' Clean the data, delete irrelevant information, delete nan entries'''
delete_field_list = ['Filename', 'catch', 'catchAns', 'subage', 'submale', 'subrace',
                     'catch.1', 'catchAns.1', 'subage.1', 'submale.1', 'subrace.1']
df.drop(delete_field_list, inplace=True, axis=1)  # shortlist
df.dropna(axis=0, how='any', inplace=True)  # delete any row that contains a nan


''' randomly divide subjects into 2 groups. calculate the average for each group '''
num_im = df['Image #'].max()
head_list = list(df.columns.values)
feature_list = head_list[1:]
num_feature = len(feature_list)

ave_1 = np.zeros((num_im, num_feature))
ave_2 = np.zeros((num_im, num_feature))

for cur_im in range(1, num_im+1):
    tmp_data = df[df['Image #'] == cur_im]
    group_1, group_2 = train_test_split(tmp_data.values[:, 1:], test_size=0.5, random_state=42)
    ave_1[cur_im-1, :], ave_2[cur_im-1, :] = group_1.mean(axis=0), group_2.mean(axis=0)


''' Now calculate the correlation for each feature '''
correlation_array = np.zeros((num_feature, 2))
for cur_feature in range(num_feature):
    correlation_array[cur_feature, :] = pearsonr(ave_1[:, cur_feature], ave_2[:, cur_feature])

for i in range(num_feature):
    if correlation_array[i, 0] > 0.4 and correlation_array[i, 1] < 0.05:
        print '%s:  %.2f' % (feature_list[i], correlation_array[i, 0])


''' plot a bar graph '''
N = num_feature
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, correlation_array[:, 0], width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Consistency')
ax.set_title('Consistency by feature')
plt.xticks(ind+width, feature_list, rotation='vertical')
plt.savefig('../figs/consistency.png')
plt.show()
