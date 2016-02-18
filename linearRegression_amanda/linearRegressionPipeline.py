import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model
# import matplotlib.pyplot as plt
# %matplotlib inline

__author__ = 'amanda'


'''
The purpose of this code is to predict attractiveness
using linear regression on the facial features (proposed by Chicago database)
'''


# Load the face feature matrix
feature_arr = pd.read_csv('allFeatures.csv')
# Remove the irrelevant information
del feature_arr['dataset']
del feature_arr['img_num']


# Do PCA to decorrelate the data
explained_variance = 0.99
pca = PCA(n_components=explained_variance)
new_featureArray = pca.fit_transform(feature_arr)
print 'The number of PCs needed to retain %.3f variance is %d.' \
      % (explained_variance, new_featureArray.shape[1])


# Load the rating array
full_rating = pd.read_csv('rating_matrix.csv', delimiter='\t')
del full_rating['Unnamed: 0']
rater_ind = 0  # Later, we will iterate over this number.

one_rating = full_rating.iloc[rater_ind, :][:, None]
one_rating = [map(int, x) for x in one_rating]
one_rating = np.array(one_rating)


# Do linear regression on feature_arr and one_rating
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(feature_arr, one_rating)
predicted_rating = regr.predict(feature_arr)
# rectified_rating = np.around(predicted_rating, decimals=0)

# Calculate the mean square error
MSE = np.mean((predicted_rating - one_rating) ** 2)
print 'Residual sum of squares: %.2f' % MSE


# Calculate how much variance is explained
variance_score = regr.score(feature_arr, one_rating)
print 'Variance score is: %.2f' % variance_score


# Calculate the correlation between prediction and actual rating.
cor = np.corrcoef(predicted_rating[:, 0], one_rating[:, 0])
print cor[0, 1]

# # # Plot prediction vs actual rating.
# x = predicted_rating
# y = one_rating
# fig, ax = plt.subplots()
# ax.scatter(x, y, alpha=0.5)
# ax.set_xlim((0, 8))
# ax.set_ylim((0, 8))
# x0, x1 = ax.get_xlim()
# y0, y1 = ax.get_ylim()
# ax.set_aspect(abs(x1-x0)/abs(y1-y0))
# ax.grid(b=True, which='major', color='k', linestyle='--')
# plt.xlabel('Predicted Ratings')
# plt.ylabel('Actual Ratings')
# plt.title('Predicted VS Actual Ratings')
# plt.savefig('scatter_1.png')

















