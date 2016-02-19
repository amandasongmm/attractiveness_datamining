import numpy as np
import pandas as pd


'''
This code converts the raw data into a user-by-image rating matrix. The saved txt file 'matrix_form.txt'
will be the raw input to the collaborative filtering codes.
'''

__author__ = 'amanda'

raw_csv_path = '../data/germine_CurrentBiology_twinaesthetics_PUBLIC.csv'
df = pd.read_csv(raw_csv_path)

# MIT 1-50, Glasgow 1-50, Others 1-50, Genhead 1-50, 200 face images in total
half1_ind = np.r_[5:55, 70:120, 200:250, 265:315]   # ratings from the first person in a twin pair
half2_ind = np.r_[399:449, 464:514, 594:644, 659:709]  # the same format, from the second person in a twin pair

df_half1 = df.iloc[:, half1_ind]
df_half2 = df.iloc[:, half2_ind]

# To rename the column of dataframe1 and dataframe2 into im1 - im200, so we can concatenate them
new_column_name = df_half1.columns.values
count = 0
for i in new_column_name:
    new_column_name[count] = 'img'+str(count+1)
    count += 1

df_half1.columns = new_column_name
df_half2.columns = new_column_name
df_new = pd.concat([df_half1, df_half2])

# save it into the same folder.
df_new.to_csv('matrix_form.csv', sep='\t')

