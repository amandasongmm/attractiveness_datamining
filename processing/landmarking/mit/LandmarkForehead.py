#
# Landmarker.py
#
# The purpose of this script is to use dlib to perform facial landmarking on
# the 200 faces which we plan to extract Chicago Dataset features from.
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Author: 	Chad Atalla
# Date: 	5/12/2016
#



import os
import sys
import dlib
from skimage import io
import pandas as pd
import numpy as np



# Set this flag to 1 in order to visualize results
VISUALIZE = 0



'''
	Label:		ParseCommandLine
	Purpose:	Parse the command line args
'''
predictor_path = 'landmarks68.dat'
faces_folder_path = '../../../MIT2kFaceDataset/2kfaces/'



'''
	Label:		SetUpDetector
	Purpose:	Use standard dlib setup for 
'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)



'''
	Label:		ProcessImages
	Purpose:	Use dlib to landmark faces, store in a dataframe
'''
# Create the dataframe for storing landmarks
landmarkMatrix = pd.DataFrame(columns=['image', 'point', 'x', 'y'])

# Track which image to landmark in the non-linear sequence
locations = pd.read_csv('../../../MIT2kFaceDataset/2kfaces/dir.csv')
locations.columns = ['paths', 'nan']

# Change constant at top to see visualizations
if (VISUALIZE == 1):
	win = dlib.image_window()

# Iterate over each image and generate + store landmarks
for imgName in locations['paths']:
	imgFile = os.path.join(faces_folder_path, imgName)
	img = io.imread(imgFile)

	x = img.shape[1]/2

	landmarkMatrix = landmarkMatrix.append(pd.Series([str(imgFile), 68, x, 0], index=['image', 'point', 'x', 'y']), ignore_index=True)


'''
	Label:		SaveLandmarks
	Purpose:	Save the dataframe to a csv for later use
'''
landmarkMatrix.to_csv('mitLandmarksForehead.csv', index=False)
print 'Finished landmarking. Check cfdLandmarks.csv'
