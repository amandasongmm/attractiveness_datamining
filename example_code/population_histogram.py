import numpy as np
import pandas as pd

__author__ = 'amanda'

# Load the rating data
tmp = np.load('tmp/clean_rating_data.npz')
orig_rating = tmp['full_rating']  # 1540 * 200

# plot a histogram of the face ratings for each dataset.
columns = ['rating', 'dataset_ind']
hist_df = pd.DataFrame(columns=columns)
for cur_dataset_ind in range(4):
    face_num = 50
    rater_num = orig_rating.shape[0]
    cur_rating = orig_rating[:, cur_dataset_ind*50:50+cur_dataset_ind*50]
    cur_rating = np.reshape(cur_rating, (face_num*rater_num, 1))
    dataset_ind = np.ones((face_num*rater_num, 1))*(cur_dataset_ind+1)
    cur_df = pd.DataFrame(np.hstack((cur_rating, dataset_ind)), columns=columns)
    hist_df = hist_df.append(cur_df, ignore_index=True)
