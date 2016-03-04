"""
LassoRegression.py

The purpose of this script is to perform lasso regression to calculate 
the weights of the different features 

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/24/2016
"""



import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import train_test_split



NUM_FEATS = 29
alpha = .00002


# Load in the original features
x = pd.read_csv('allFeatures.csv', usecols=range(2,31))
x = preprocessing.scale(x)

# Load in everyone's ratings for the 200 faces
yData = pd.read_csv('ratingMatrixChad.csv')

# Split ratings into test and train set
yTrain, yTest = train_test_split(yData, train_size=.8)

# Dataframe to hold the scores of each linear model
varScore = pd.Series(yTrain.index.size)

# Iterate over each person's 200 ratings
i = 0
for row, index in yTrain.iterrows():
    # Take one person's ratings at a time
    y = index.as_matrix()

    # Run Linear regression
    myModel = linear_model.Lasso(alpha=alpha, fit_intercept=True, copy_X=True, normalize=True)
    myModel.fit(x,y)

    # Save the calculated variance scores
    k = 0
    total = 0
    for j, vals in yTest.iterrows():
        total += myModel.score(x,vals)
        k += 1
    total /= k

    print myModel.score(x,y)
    varScore.loc[i] = total
    print total

    i += 1

varScore.to_csv(('varianceScore' + str(alpha) + '.csv'), index=False)
