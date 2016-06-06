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
def crossVal(mean_rating, featureMat, pModel = prediction_model, valSize= 0.1, MODEL= 'config',printResult = False):
    # using cross validation to find the optimal number of features
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
        featureMat, mean_rating, random_state=0, test_size=valSize)
    corrList = []
    varList = []
    mseList = []
    if MODEL != 'faceSNN':
        numFeature = np.linspace(20,500,num = 50,dtype = np.int16)
    else:
        numFeature = np.linspace(0,50,num = 10,dtype = np.int16 )
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
    if printResult:
        print 'Correlation: ', max(corrList)
        print 'num of features: ', numFeature[np.argmax(corrList)]
        print 'R^2 score: ', max(varList)
        print 'num of features: ', numFeature[np.argmax(varList)]
        print 'MSE: ', min(mseList)
        print 'num of features: ', numFeature[np.argmin(mseList)]
    optNumFea = numFeature[np.argmax(corrList)]
    X_train_hat = X_train[:, :optNumFea]
    return X_train_hat, optNumFea


def Train_Test(mean_rating, featureMat, pModel = prediction_model, hyperParam = None, xVal = False,\
               numTrain = 20,savePath = '../Result',MODEL= 'config', printToFile = False,ratio = 0.5,\
               returnValTrain = False,returnModel = False, printResult = True,getMaxMin = False):
    if hyperParam != None :
        featureMat_hat = featureMat[:, :hyperParam]
    else:
        featureMat_hat = featureMat
    dataLen = featureMat_hat.shape[0]
    corrValiList = []
    varValiList = []
    mseValiList = []
    corrTrainList = []
    varTrainList = []
    mseTrainList = []
    index_list = range(dataLen)
    for i in range(numTrain):
        index_random  = index_list
        random.shuffle(index_random)
        train_index = index_random[:int(math.ceil(float(dataLen)*ratio))]
        test_index = index_random[int(math.ceil(float(dataLen)*ratio)):]
        feature_train, feature_test = featureMat_hat[train_index], featureMat_hat[test_index]
        rating_train, rating_test = mean_rating[train_index], mean_rating[test_index]
        if hyperParam == None and xVal:
            _, optNumFea = crossVal(rating_train,feature_train,pModel = pModel, MODEL = MODEL)
            feature_train = feature_train[:,:optNumFea]
            feature_test = feature_test[:,:optNumFea]
        # Do linear regression on feature_arr and mean_rating
        pModel.fit(feature_train, rating_train)
        
        ######### on validation set #############
        predicted_rating = pModel.predict(feature_test)
        # Calculate the mean square error
        MSE = np.mean((predicted_rating - rating_test) ** 2)
        mseValiList.append(MSE)
        # Returns the coefficient of determination R^2 of the prediction.
        variance_score = pModel.score(feature_test, rating_test)
        varValiList.append(variance_score)
        # Calculate the correlation between prediction and actual rating.
        cor = np.corrcoef(predicted_rating, rating_test)
        corrValiList.append(cor[0, 1])
        
        ############# on train set ############
        predicted_rating_train = pModel.predict(feature_train)
        # Calculate the mean square error
        MSE = np.mean((predicted_rating_train - rating_train) ** 2)
        mseTrainList.append(MSE)
        # Returns the coefficient of determination R^2 of the prediction.
        variance_score = pModel.score(feature_train, rating_train)
        varTrainList.append(variance_score)
        # Calculate the correlation between prediction and actual rating.
        cor = np.corrcoef(predicted_rating_train, rating_train)
        corrTrainList.append(cor[0, 1])
    if not xVal: 
        optNumFea = hyperParam
    if printResult:
        print '**************************Result of train and test**************************************'
        print 'number of features: %d' % optNumFea
        print 'On validation set:'
        print 'Residual sum of squares: %.2f' % (sum(mseValiList) / numTrain)
        print 'Variance score is: %.2f' % (sum(varValiList) / numTrain)
        print 'Correlation between predicted ratings and actual ratings is: %.4f' % (sum(corrValiList) / numTrain)
        print ' '
        print 'On training set:'
        print 'Residual sum of squares: %.2f' % (sum(mseTrainList) / numTrain)
        print 'Variance score is: %.2f' % (sum(varTrainList) / numTrain)
        print 'Correlation between predicted ratings and actual ratings is: %.4f' % (sum(corrTrainList) / numTrain)
        print '****************************************************************************************'

    if printToFile:
        fName = savePath + '/' + MODEL + '_trainTest.txt'
        with open(fName, 'w') as f:
            f.write('Number of train/test: %d' % numTrain + '\n')
            f.write('Residual sum of squares: %.2f' % (sum(mseTrainList) / numTrain) + '\n')
            f.write('Variance score is: %.2f' % (sum(varTrainList) / numTrain) + '\n')
            f.write('Correlation between predicted ratings and actual ratings is: %.4f' \
                    % (sum(corrTrainList) / numTrain) + '\n')
            f.write('\n')
            f.write('Residual sum of squares: %.2f' % (sum(mseValiList) / numTrain) + '\n')
            f.write('Variance score is: %.2f' % (sum(varValiList) / numTrain) + '\n')
            f.write('Correlation between predicted ratings and actual ratings is: %.4f' \
                    % (sum(corrValiList) / numTrain) + '\n')
    if getMaxMin:
        maxRating,maxIndex,minRating,minIndex = getPredResult(predicted_rating,test_index)
        return maxRating,maxIndex,minRating,minIndex
    
    if returnModel:
        return pModel
    
    if returnValTrain:
        return (sum(corrValiList) / numTrain), (sum(corrTrainList) / numTrain)
    
def getPredResult(predicted_rating,test_index):
    #print test_index
    test_index = np.asarray(test_index)
    copy_rating = np.asarray(predicted_rating)
    maxIndex = test_index[copy_rating.argsort()[-5:][::-1]]
    minIndex = test_index[copy_rating.argsort()[:5]]
    copy_rating.sort()
    maxRating = copy_rating[-5:]
    minRating = copy_rating[:5]
    print '**************************Result of predicted max and min on testing set****************'
    print 'maxIndex: ',maxIndex
    print 'minIndex: ',minIndex
    print 'maxRating: ',maxRating
    print 'minRating: ',minRating
    print '****************************************************************************************'
    return maxRating,maxIndex,minRating,minIndex

