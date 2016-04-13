"""
TwinSetPCA.py

The purpose of this script is to run PCA on the features from the twin set

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/17/2016
"""



import pandas as pd
import numpy as np
from sklearn.decomposition import PCA



#
# Label:   LoadData
# Purpose: Load in the twin set feature matrix
#
data = pd.read_csv('./scaledFeatures.csv')
#data = pd.read_csv('./landmarking/allFeatures.csv', usecols=range(2,31))
npData = data.as_matrix()



#
# Label:   RunPCA
# Purpose: Run PCA on the features to determine dimensionality
#
pca = PCA(n_components=.95)
newFeatures = pca.fit_transform(npData)

# Put the new feature matrix into a dataframe
newData = pd.DataFrame(newFeatures)

# Calculate the primary component compositions
allPC = PCA()
allPCs = allPC.fit_transform(npData)
identity = np.identity(29) # For 29 initial features
composition = pd.DataFrame(allPC.transform(identity))



#
# Label:   PrintAndSave
# Purpose: Print and save PCA results and new features
#
print '\nvariance ratios = '
print pca.explained_variance_ratio_
print '\nn_components = '
print pca.n_components_

newData.to_csv('PCA_Features.csv', index=False)
composition.to_csv('PCCompositions.csv', index=False)