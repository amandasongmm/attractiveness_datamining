"""
OneDim.py

The purpose of this script is to plot a pc in one dimension

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/6/2016
"""

CHOICE = 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load up the calculated PC's for the 200 images
# Order is MIT, gs, ngs, gh
pcs = pd.read_csv('PCA_Features.csv')

# Base path to images
basePath = '../../imageProcessing/paddedImages/cropped'

def main():
    # Get full paths to all images
    image_paths = [basePath + str(k) + '.png' for k in range(200)]

    # First run for PC 1 vs PC 2
    z1 = CHOICE

    # First run for all images
    imgSet = "All Images"
    imgs = image_paths

     # Plot PC1 vs PC2
    x = pcs.iloc[:, (z1-1)].values
    y = [1] * len(x)
    genPlot(x, y, imgs, z1, imgSet)

    # Switch to only subsets
    imgSet = 'MIT'
    genPlot(x[0:49], y[0:49], imgs[0:49], z1, imgSet)
    imgSet = 'GS'
    genPlot(x[50:99], y[50:99], imgs[50:99], z1, imgSet)
    imgSet = 'NGS'
    genPlot(x[100:149], y[100:149], imgs[100:149], z1, imgSet)
    imgSet = 'GH'
    genPlot(x[150:199], y[150:199], imgs[150:199], z1, imgSet)

def genPlot(x, y, imgs, z1, imgSet):


    # Prepare the plot for putting thumbnails
    fig, ax = plt.subplots()
    plt.xlabel('PC ' + str(z1))
    fig.suptitle('PC ' + str(z1) + ', ' + imgSet)

    # Add thumbnails
    imscatter(x, y, imgs, zoom=0.3, ax=ax)

    # Make base scatter to double check
    ax.scatter(x, y)
    plt.show()

def imscatter(x, y, imgs, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    # Make sure they are Numpy arrays
    x, y = np.atleast_1d(x, y)

    # Used to add faces to plot
    thumbnails = []

    # Add image overlays one by one
    i = 0
    for x0, y0 in zip(x, y):
        image = plt.imread(imgs[i])
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        thumbnails.append(ax.add_artist(ab))
        i += 1

    # Scale plot based on added images
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    
    return thumbnails

# Run the process
main()