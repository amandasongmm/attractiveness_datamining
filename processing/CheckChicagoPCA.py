"""
CheckChicagoPCA.py

The purpose of this script is to perform PCA on the Chicago Features in order
to determine if their dimensionality is similar to ours

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/15/2016
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA



#
# Label:   LoadData
# Purpose: Load the data from memory
#
data = pd.read_csv('../ChicagoFaceDataset/cfd2Data.csv', usecols=(range(15, 69)))



#
# Label:   RemoveUnused
# Purpose: Remove unused columns
#
data.drop(['Babyface', 'Disgusted', 'Dominant', 'Feminine', 'Happy', 'Masculine', 'Prototypic', 'Sad', 'Suitability', 'Surprised', 'Threatening', 'Trustworthy', 'Unusual', 'Luminance_median', 'L_Eye_H', 'R_Eye_H', 'L_Eye_W', 'R_Eye_W', 'Asymmetry_pupil_top', 'Asymmetry_pupil_lip', 'Midcheek_Chin_L', 'Midcheek_Chin_R'], axis=1, inplace=True)



#
# Label:   AverageLR
# Purpose: Average the L and R measurements and remove the L & R cols
#
data['Midbrow_Hairline'] = range(597)
data['Pupil_Lip'] = range(597)
data['Pupil_Top'] = range(597)

i = 0
for index, row in data.iterrows(): 
    data['Midbrow_Hairline'][i] = (int(row['Midbrow_Hairline_L']) + int(row['Midbrow_Hairline_R']))/2
    i = i + 1
data.drop(['Midbrow_Hairline_L', 'Midbrow_Hairline_R'], axis=1, inplace=True)

for index, row in data.iterrows(): 
    data['Pupil_Top'][i] = (int(row['Pupil_Top_L']) + int(row['Pupil_Top_R']))/2
    i = i + 1
data.drop(['Pupil_Top_R', 'Pupil_Top_L'], axis=1, inplace=True)

for index, row in data.iterrows(): 
    data['Pupil_Lip'][i] = (int(row['Pupil_Lip_L']) + int(row['Pupil_Lip_R']))/2
    i = i + 1
data.drop(['Pupil_Lip_R', 'Pupil_Lip_L'], axis=1, inplace=True)



#
# Label:   RunPCA
# Purpose: Run PCA on the features to determine dimensionality
#
forPCA = data.drop('Attractive', axis=1).as_matrix()
pca = PCA(n_components=.95)
pca.fit(forPCA)

print '\nvariance ratios = '
print pca.explained_variance_ratio_
print '\nn_components = '
print pca.n_components_