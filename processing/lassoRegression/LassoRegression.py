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
TRIALS = 5          # Num of test / train splits to test on
alpha = .00002


# Load in the original features
x = pd.read_csv('allFeatures.csv', usecols=range(2,31))
x = preprocessing.scale(x)

# Load in everyone's ratings for the 200 faces
yData = pd.read_csv('ratingMatrixChad.csv')

# Dataframe to hold the scores of each linear model
varScore = pd.Series(yData.index.size)
correlation = pd.Series(yData.index.size)
MSEs = pd.Series(yData.index.size)

# Iterate over each person's ratings
i = 0
for row, index in yData.iterrows():
    # Take one person's ratings at a time
    y = index.as_matrix()

    # Hold aggregate model data for different test / train splits
    scores = pd.Series(TRIALS)
    mses = pd.Series(TRIALS)
    predictions = pd.Series()
    yVals = pd.Series()

    # Do multiple train / test splits
    for j in range(TRIALS):
        # Split ratings into test and train set
        yTrain, yTest, xTrain, xTest = train_test_split(y, x, train_size=.99)

        # Run Lasso regression
        myModel = linear_model.Lasso(alpha=alpha, fit_intercept=True, copy_X=True, normalize=True)
        myModel.fit(xTrain,yTrain)

        scores.loc[j] = myModel.score(xTest, yTest)

        # Save the predicted ratings
        prediction = myModel.predict(xTest)
        predictions = predictions.append(pd.Series(prediction))
        yVals = yVals.append(pd.Series(yTest))
        mses.loc[j] = np.mean((prediction - yTest) ** 2)

    # Save the calculated variance score
    varScore.loc[i] = np.mean(scores)

    # Save the MSE vals
    MSEs.loc[i] = np.mean(mses)

    # Save the correlation
    correlation.loc[i] = (np.corrcoef(predictions, yVals)[0][1])

    i += 1

MSEs.to_csv(('mse' + str(alpha) + '.csv'), index=False)
varScore.to_csv(('varianceScore' + str(alpha) + '.csv'), index=False)
correlation.to_csv(('correlation' + str(alpha) + '.csv'), index=False)
