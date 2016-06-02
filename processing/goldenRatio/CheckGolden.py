#
# CheckGolden.py
#
# The purpose of this script is to determine the ratios of face positions
# from this video: https://www.youtube.com/watch?v=kKWV-uU_SoI 
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Author: 	Chad Atalla
# Date: 	2/3/2016
#



import pandas as pd
import numpy as np
from scipy.spatial import distance



NUM_IMG = 15
NUM_LANDMARK = 68
GOLDEN = .618



'''
	Label:		ParseLandmarks
	Purpose:	Parse the landmarks from the csv
'''
data = pd.read_csv('allLandmarks.csv')



'''
	Label:		CreateDataFrame
	Purpose:	Create a dataframe with the structure for storing featuress
'''
features = pd.DataFrame(columns=('img_name', 'tier', 'residual1', 'residual2', 'residual3', 'residual4', 'residual5'))



'''
	Label:		LoopOnImages
	Purpose:	Begin loop that will easily allow feature extraction from each
'''
# Loop over all landmarks for each of the 200 images
for i in range(NUM_IMG):
	# Get current file name
	curName = (data.loc[i*NUM_LANDMARK])['image']
	tier = i / 5

	# Create a list where all landmarks can be temporarily stored for this img
	curLandmarks = []

	# Loop over each landmark for this image
	for j in range(NUM_LANDMARK):
		# Add the current points to the curLandmarks list
		row = data[data.image == curName][data.point == j]
		curLandmarks.append([int(row['x']), int(row['y'])])

	# curLandmarks is now a list of x,y tuples of landmarks for this image



	'''
		Label:		CalculateFeatures
		Purpose:	Calculate features from the list of landmarks
	'''

	# ratio 1 = #2 from the video
	ratio1_1 = distance.euclidean(
		np.mean([curLandmarks[36], curLandmarks[45]], axis=0),
		np.mean([curLandmarks[31], curLandmarks[35]], axis=0))

	ratio1_2 = distance.euclidean(
		np.mean([curLandmarks[36], curLandmarks[45]], axis=0),
		np.mean([curLandmarks[61], curLandmarks[63]], axis=0))

	ratio1 = ratio1_1 / ratio1_2
	d1 = abs(GOLDEN - ratio1)

	# ratio 2 = #3 from the video
	ratio2_1 = distance.euclidean(
		np.mean([curLandmarks[36], curLandmarks[45]], axis=0),
		curLandmarks[33])

	ratio2_2 = distance.euclidean(
		np.mean([curLandmarks[36], curLandmarks[45]], axis=0),
		curLandmarks[57])

	ratio2 = ratio2_1 / ratio2_2
	d2 = abs(GOLDEN - ratio2)

	
	# ratio 3 = #4 from the video
	ratio3_1 = distance.euclidean(
		np.mean([curLandmarks[36], curLandmarks[45]], axis=0),
		np.mean([curLandmarks[61], curLandmarks[63]], axis=0))

	ratio3_2 = distance.euclidean(
		np.mean([curLandmarks[36], curLandmarks[45]], axis=0),
		curLandmarks[8])

	ratio3 = ratio3_1 / ratio3_2
	d3 = abs(GOLDEN - ratio3)

	# ratio 4 = #7 from the video
	ratio4_1 = distance.euclidean(
		np.mean([curLandmarks[50], curLandmarks[52]], axis=0),
		curLandmarks[57])

	ratio4_2 = distance.euclidean(
		np.mean([curLandmarks[50], curLandmarks[52]], axis=0),
		curLandmarks[8])

	ratio4 = ratio4_1 / ratio4_2
	d4 = abs(GOLDEN - ratio4)


	# ratio 5 = #14 from the video
	ratio5_1 = distance.euclidean(
		curLandmarks[0],
		curLandmarks[39])

	ratio5_2 = distance.euclidean(
		curLandmarks[0],
		curLandmarks[42])

	ratio5 = ratio5_1 / ratio5_2
	d5 = abs(GOLDEN - ratio5)


	'''
		Label:		SaveFeatures
		Purpose:	Saves the features as a row entry in a dataframe
	'''
	curFeatures = pd.Series([curName, tier, d1, d2, d3, d4, d5], index=features.columns)

	features.loc[i] = curFeatures

print features
features.to_csv('residuals.csv', index=False)