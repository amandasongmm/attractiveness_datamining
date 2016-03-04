"""
CropImages.py

The purpose of this script is to crop the faces from the different data sets 
and normalize their height

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/27/2016
"""

from PIL import Image
import pandas as pd
import os

faces_folder_path = '../../imageData/'

# Load in the landmark points of the faces
landmarks = pd.read_csv('allOriginalLandmarks.csv')

# Track the order of features in the 4 datasets
fileLocs = pd.read_csv('imageLocations.csv')

# Iterate over each image and crop
for imgNum in range(200):
    # Get current file name
    curName = (landmarks.loc[imgNum*69])['image']

    # Open the image file
    imgFile = os.path.join(faces_folder_path, fileLocs['locations'][imgNum])
    img = Image.open(imgFile)

    # Create a list where the bounding points can be stored
    bounds = []

    # Fill the bounds for this image (left, top, right, bottom)
    for j in [0, 68, 16, 8]:
        # Add the current points to the bounds list
        row = landmarks[landmarks.image == curName][landmarks.point == j]
        bounds.append([int(row['x']), int(row['y'])])

    # Save the X or Y values only from the bounds
    bounds = [bounds[0][0], bounds[1][1], bounds[2][0], bounds[3][1]]

    # Crop the images to perfectly fit the face
    img = img.crop(bounds)

    # Shrink ratio
    ratio = 200/float(img.size[1])
    newWidth = int(img.size[0]*ratio)

    # Scale all images to the min height which is 200
    img = img.resize((newWidth, 200))

    # Save the modified image
    img.save('./croppedImages/cropped' + str(imgNum) + ".png")