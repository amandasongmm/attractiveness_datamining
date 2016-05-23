#
# ExtractFeatures.py
#
# The purpose of this script is to extract the Chicago Dataset features
# from the landmarks of the faces from the MIT, GS, NGS, and GEN sets
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Author: 	Chad Atalla
# Date: 	2/3/2016
#



import pandas as pd
import numpy as np
from scipy.spatial import distance



NUM_IMG = 2
NUM_LANDMARK = 69



'''
	Label:		ParseLandmarks
	Purpose:	Parse the landmarks from the csv
'''
data = pd.read_csv('landmarks.csv')



'''
	Label:		CreateDataFrame
	Purpose:	Create a dataframe with the structure for storing featuress
'''
features = pd.DataFrame(columns=('dataset', 'img_num', 'nose_width', 'nose_length', 'lip_thickness', 'face_length', 'eye_height', 'eye_width', 'face_width_prom', 'face_width_mouth', 'forehead_length', 'distance_btw_pupils', 'dist_btw_pupils_top','dist_btw_pupils_lip', 'chin_length', 'length_cheek_to_chin', 'brow_to_hair', 'fWHR', 'face_shape', 'heartshapeness', 'nose_shape', 'lip_fullness', 'eye_shape', 'eye_size', 'upper_head_len', 'midface_len', 'chin_size', 'forehead_height', 'cheek_height', 'cheek_prominence', 'face_roundness'))



'''
	Label:		LoopOnImages
	Purpose:	Begin loop that will easily allow feature extraction from each
'''
# Loop over all landmarks for each of the 200 images
for i in range(NUM_IMG):
	# Get current file name
	curName = (data.loc[i*NUM_LANDMARK])['image']
	dataset = curName[17: curName.find('/', 18)]

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
	# These calculations are based off of data provided in this directory
	# regarding landmark positions and Chicago feature calculations
	nose_width = distance.euclidean(curLandmarks[35], curLandmarks[31])

	nose_length = distance.euclidean(curLandmarks[33], curLandmarks[27])

	lip_thickness = (distance.euclidean(curLandmarks[50], curLandmarks[61]) + distance.euclidean(curLandmarks[52], curLandmarks[63]) + 2*distance.euclidean(curLandmarks[57], curLandmarks[66]))/2

	face_length = distance.euclidean(curLandmarks[8], curLandmarks[68])

	eye_height_r = distance.euclidean(np.mean([curLandmarks[37], curLandmarks[38]], axis=0), np.mean([curLandmarks[40], curLandmarks[41]], axis=0))
	eye_height_l = distance.euclidean(np.mean([curLandmarks[43], curLandmarks[44]], axis=0), np.mean([curLandmarks[47], curLandmarks[46]], axis=0))
	eye_height = (eye_height_l + eye_height_r)/2

	eye_width = (distance.euclidean(curLandmarks[36], curLandmarks[39]) + distance.euclidean(curLandmarks[42], curLandmarks[45]))/2

	face_width_prom = distance.euclidean(curLandmarks[1], curLandmarks[15])

	face_width_mouth = distance.euclidean(curLandmarks[4], curLandmarks[12])

	forehead_length = distance.euclidean(curLandmarks[27], curLandmarks[68])

	distance_btw_pupils = distance.euclidean(np.mean([curLandmarks[37], curLandmarks[38]], axis=0), np.mean([curLandmarks[43], curLandmarks[44]], axis=0))

	pupil_center = np.mean([np.mean([curLandmarks[37], curLandmarks[40]], axis=0), np.mean([curLandmarks[44], curLandmarks[47]], axis=0)], axis=0)
	dist_btw_pupils_top =  distance.euclidean(pupil_center, curLandmarks[68])

	dist_btw_pupils_lip = (distance.euclidean(np.mean([curLandmarks[37], curLandmarks[40]], axis=0), curLandmarks[49]) + distance.euclidean(np.mean([curLandmarks[44], curLandmarks[47]], axis=0), curLandmarks[53]))/2

	chin_length = distance.euclidean(curLandmarks[57], curLandmarks[8])

	length_cheek_to_chin = (distance.euclidean(curLandmarks[2], curLandmarks[8]) + distance.euclidean(curLandmarks[14], curLandmarks[8]))/2

	brow_to_hair = distance.euclidean(np.mean([curLandmarks[19], curLandmarks[24]], axis=0), curLandmarks[68])

	lip_to_brow = distance.euclidean(np.mean([curLandmarks[19], curLandmarks[24]], axis=0), curLandmarks[51])
	fWHR = face_width_prom / lip_to_brow

	face_shape = face_width_prom / face_length

	heartshapeness = face_width_prom / face_width_mouth

	nose_shape = nose_width / nose_length

	lip_fullness = lip_thickness / face_length

	eye_shape = eye_height / eye_width

	eye_size = eye_height / face_length

	upper_head_len = forehead_length / face_length

	midface_len = dist_btw_pupils_lip / face_length

	chin_size = chin_length / face_length

	forehead_height = brow_to_hair / face_length

	cheek_height = length_cheek_to_chin / face_length

	cheek_prominence = (face_width_prom - face_width_mouth) / face_length

	face_roundness = face_width_mouth / face_length



	'''
		Label:		SaveFeatures
		Purpose:	Saves the features as a row entry in a dataframe
	'''
	curFeatures = pd.Series([dataset, i%50, nose_width, nose_length, lip_thickness, face_length, eye_height, eye_width, face_width_prom, face_width_mouth, forehead_length, distance_btw_pupils, dist_btw_pupils_top,dist_btw_pupils_lip, chin_length, length_cheek_to_chin, brow_to_hair, fWHR, face_shape, heartshapeness, nose_shape, lip_fullness, eye_shape, eye_size, upper_head_len, midface_len, chin_size, forehead_height, cheek_height, cheek_prominence, face_roundness], index=features.columns)

	features.loc[i] = curFeatures

features.to_csv('allFeatures.csv', index=False)