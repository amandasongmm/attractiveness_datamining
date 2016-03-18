"""
LinearRegression.py

The purpose of this script is to perform linear regression to calculate 
the weights of the different primary components from PCA on the twin set

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   3/17/2016
"""



import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


NUM_PCS = 11
TRIALS = 5


# Load in the 6 PCs for image features
x = pd.read_csv('PCA_Features.csv')
x = preprocessing.scale(x)

# Load in everyone's ratings for the 200 faces
yData = pd.read_csv('ratingMatrixChad.csv')

# Dataframe to hold the coefficients calculated for each person's linregress
predicted = pd.DataFrame(columns=range(1,201))
coefficients = pd.DataFrame(columns=range(NUM_PCS))
MSEs = pd.Series(range(yData.index.size))
varScore = pd.Series(yData.index.size)
correlation = pd.Series(yData.index.size)

# Iterate over each person's 200 ratings
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

        # Run Linear regression
        myModel = linear_model.LinearRegression(fit_intercept=True, copy_X=True, normalize=True)
        myModel.fit(x,y)

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

varScore.to_csv('varianceScore.csv', index=False)
MSEs.to_csv('mseVals.csv', index=False)
correlation.to_csv('correlations.csv', index=False)