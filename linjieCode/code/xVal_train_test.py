'''
using ridge regression.
Please change the prediction_model if you wish to use other linear models
'''
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
import random
import math

prediction_model = linear_model.Ridge(fit_intercept=True)
def crossVal(mean_rating, featureMat, pModel = prediction_model, valSize= 0.1, MODEL= 'config'):
    # using cross validation to find the optimal number of features
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
        featureMat, mean_rating, random_state=0, test_size=valSize)
    corrList = []
    varList = []
    mseList = []
    if MODEL != 'faceSNN':
        numFeature = [40, 50, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200, 250, 300, 350]
    else:
        numFeature = [10, 20, 30, 40, 50]
    for numF in numFeature:
        X_train_hat = X_train[:, :numF]
        X_test_hat = X_test[:, :numF]
        pModel.fit(X_train_hat, y_train)
        predicted_rating = pModel.predict(X_test_hat)
        # Calculate the mean square error
        MSE = np.mean((predicted_rating - y_test) ** 2)
        mseList.append(MSE)
        # Returns the coefficient of determination R^2 of the prediction.
        '''
        The coefficient R^2 is defined as (1 - u/v),
        where u is the regression sum of squares ((y_true - y_pred) ** 2).sum()
        and v is the residual sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y,
        disregarding the input features, would get a R^2 score of 0.0.
        '''
        variance_score = pModel.score(X_test_hat, y_test)
        varList.append(variance_score)

        # Calculate the correlation between prediction and actual rating.
        cor = np.corrcoef(predicted_rating, y_test)
        corrList.append(cor[0, 1])

    print 'Correlation: ', max(corrList)
    print 'num of features: ', numFeature[np.argmax(corrList)]
    print 'R^2 score: ', max(varList)
    print 'num of features: ', numFeature[np.argmax(varList)]
    print 'MSE: ', min(mseList)
    print 'num of features: ', numFeature[np.argmin(mseList)]
    optNumFea = numFeature[np.argmax(corrList)]
    X_train_hat = X_train[:, :optNumFea]
    return X_train_hat, optNumFea


def Train_Test(mean_rating, featureMat, pModel = prediction_model, hyperParam = None,\
               numTrain = 20,savePath = '../Result',MODEL= 'config', printToFile = False):
    if hyperParam != None :
        featureMat_hat = featureMat[:, :hyperParam]
    else:
        featureMat_hat = featureMat
    dataLen = featureMat_hat.shape[0]

    corrList = []
    varList = []
    mseList = []
    index_list = range(dataLen)
    for i in range(1, numTrain):
        index_random = random.shuffle(index_list)
        train_index = index_random[:math.ceil(float(dataLen)/2)]
        test_index = index_random[math.ceil(float(dataLen)/2):]
        feature_train, feature_test = featureMat_hat[train_index], featureMat_hat[test_index]
        rating_train, rating_test = mean_rating[train_index], mean_rating[test_index]
        if hyperParam != None :
            feature_train, optNumFea = crossVal(rating_train,feature_train,pModel = pModel, MODEl = MODEL)
            feature_test = feature_test[:,:optNumFea]
        # Do linear regression on feature_arr and mean_rating
        pModel.fit(feature_train, rating_train)
        predicted_rating = pModel.predict(feature_test)

        # Calculate the mean square error
        MSE = np.mean((predicted_rating - rating_test) ** 2)
        mseList.append(MSE)

        # Returns the coefficient of determination R^2 of the prediction.
        variance_score = pModel.score(feature_test, rating_test)
        varList.append(variance_score)

        # Calculate the correlation between prediction and actual rating.
        cor = np.corrcoef(predicted_rating, rating_test)
        corrList.append(cor[0, 1])

    print 'Residual sum of squares: %.2f' % (sum(mseList) / numTrain)
    print 'Variance score is: %.2f' % (sum(varList) / numTrain)
    print 'Correlation between predicted ratings and actual ratings is: %.4f' % (sum(corrList) / numTrain)

    if printToFile:
        fName = savePath + '/' + MODEL + '_kFold.txt'
        with open(fName, 'w') as f:
            f.write('Number of train/test: %d' % numTrain + '\n')
            f.write('Residual sum of squares: %.2f' % (sum(mseList) / numTrain) + '\n')
            f.write('Variance score is: %.2f' % (sum(varList) / numTrain) + '\n')
            f.write('Correlation between predicted ratings and actual ratings is: %.4f' \
                    % (sum(corrList) / numTrain) + '\n')
