"""
PCPlots.py

The purpose of this script is to plot pc1 vs pc2, pc3 vs pc4
in order to visualize the encoded information

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/01/2016
"""

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
    z1 = 1
    z2 = 2

    # First run for all images
    imgSet = "All Images"
    imgs = image_paths

    # Plot PC1 vs PC2
    x = pcs.iloc[:, 1].values
    y = pcs.iloc[:, 2].values
    genPlot(x, y, imgs, z1, z2, imgSet)

    # Plot PC1 vs PC2
    x = pcs.iloc[:, 3].values
    y = pcs.iloc[:, 4].values
    z1 = 3
    z2 = 4
    genPlot(x, y, imgs, z1, z2, imgSet)


def genPlot(x, y, imgs, z1, z2, imgSet):


    # Prepare the plot for putting thumbnails
    fig, ax = plt.subplots()
    plt.xlabel('PC ' + str(z1))
    plt.ylabel('PC ' + str(z2))
    fig.suptitle('PC ' + str(z1) + ' vs PC ' + str(z2) + ', ' + imgSet)

    # Add thumbnails
    imscatter(x, y, imgs, zoom=0.1, ax=ax)

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