"""
LassoRegression.py

The purpose of this script is to perform lasso regression to calculate 
the weights of the different features 

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   3/30/2016
"""



import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import train_test_split



NUM_FEATS = 29
TRIALS = 3


# Load in the original features
x = pd.read_csv('allFeatures.csv', usecols=range(2,31))
x = preprocessing.scale(x)

# Load in everyone's ratings for the 200 faces
yData = pd.read_csv('ratingMatrixChad.csv')

# Hold scores of different alphas
alphaScores = []

# Loop over different alpha values
for alpha in {0, .00000000002}:

    # Dataframe to hold the scores of each linear model
    correlation = pd.Series(yData.index.size)

    # Iterate over each person's ratings
    i = 0
    for row, index in yData.iterrows():
        # Take one person's ratings at a time
        y = index.as_matrix()

        # Run Lasso regression
        myModel = linear_model.Lasso(alpha=alpha, fit_intercept=True, copy_X=True, normalize=True)
        myModel.fit(x,y)

        # Save the predicted ratings
        prediction = myModel.predict(x)

        # Save the correlation
        correlation.loc[i] = (np.corrcoef(prediction, y)[0][1])

        i += 1

    score = np.mean(correlation)
    alphaScores.append({score, alpha})

myScores = alphaScores.sort()
print alphaScores