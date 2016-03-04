"""
LinearRegression.py

The purpose of this script is to perform linear regression to calculate 
the weights of the different primary components from PCA on the twin set

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/18/2016
"""



import pandas as pd
import numpy as np
import math
from sklearn import linear_model


NUM_PCS = 11


# Load in the 6 PCs for image features
x = pd.read_csv('PCA_Features.csv')
x = x.as_matrix()

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

    # Run Linear regression
    myModel = linear_model.LinearRegression(fit_intercept=True, copy_X=True, normalize=True)
    myModel.fit(x,y)

    # Save the calculated coefficients
    coefficients.loc[i] = myModel.coef_

    # Save the calculated variance scores
    varScore.loc[i] = myModel.score(x,y)

    # Save the calculated MSE
    prediction = myModel.predict(x)
    MSEs.loc[i] = np.mean((prediction - y) ** 2)

    # Save the predicted ratings
    predicted.loc[i] = prediction

    # Save the correlation
    correlation.loc[i] = (np.corrcoef(prediction, y))[0][1]
    if math.isnan(correlation.loc[i]):
        correlation.loc[i] = 0

    i = i + 1

coefficients.to_csv('linearCoefficients.csv', index=False)
varScore.to_csv('varianceScore.csv', index=False)
MSEs.to_csv('mseVals.csv', index=False)
predicted.to_csv('predictedRatings.csv', index=False)
correlation.to_csv('correlations.csv', index=False)