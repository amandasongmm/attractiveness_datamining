#
# Landmarker.py
#
# The purpose of this script is to check vs chicago dataset
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Author: 	Chad Atalla
# Date: 	4/6/2016
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
faces_folder_path = './'



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
locations = ['200.jpg', '201.jpg']

# Change constant at top to see visualizations
if (VISUALIZE == 1):
	win = dlib.image_window()

# Iterate over each image and generate + store landmarks
for imgNum in range(len(locations)):
	imgFile = os.path.join(faces_folder_path, locations[imgNum])
	img = io.imread(imgFile)

	# Change constant at top to see visualizations
	if (VISUALIZE == 1):
		win.clear_overlay()
		win.set_image(img)

    # Detect the bounding box of the face
	box = detector(img, 1)

    # Find landmarks in the bounding box
	shape = predictor(img, box[0])

	# Change constant at top to see visualizations
	if (VISUALIZE == 1):
		win.add_overlay(shape)
		win.add_overlay(box)
		dlib.hit_enter_to_continue()

    # Add the landmarks to the dataframe
	for i in range(68):
		x = shape.part(i).x
		y = shape.part(i).y
		landmarkMatrix = landmarkMatrix.append(pd.Series([str(imgFile), i, x, y], index=['image', 'point', 'x', 'y']), ignore_index=True)



'''
	Label:		SaveLandmarks
	Purpose:	Save the dataframe to a csv for later use
'''
landmarkMatrix.to_csv('allLandmarksWithoutHairline.csv', index=False)
print 'Finished landmarking. Check allLandmarks.csv'
