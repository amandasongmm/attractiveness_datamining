"""
PadImages.py

The purpose of this script is to pad the images so they can be passed into a
neural net (to learn features)

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   3/6/2016
"""

import matplotlib.pyplot as plt
import numpy as np


# Iterate over each image and add padding
for imgNum in range(200):
    # Get current file name
    imgFile = ('./croppedImages/cropped' + str(imgNum) + ".png")

    # Open the image file
    img = plt.imread(imgFile)

    # Create a back canvas of the padding
    imgNew = np.zeros((200, 200, 3), dtype=img.dtype)

    # Figure out how much padding goes on each side
    padding = 200 - img.shape[1]
    leftPadding = padding/2
    rightPadding = padding - leftPadding

    # Paste the image into the padding background, centered
    imgNew[:,leftPadding:-rightPadding,:] = img
    
    # Save the padded image
    plt.imsave(('./paddedImages/cropped' + str(imgNum) + ".png"), imgNew)
