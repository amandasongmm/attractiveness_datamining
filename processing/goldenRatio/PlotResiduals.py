"""
PlotResiduals.py

The purpose of this script is to plot residuals from golden ratio checks

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   6/01/2016
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

NUMIMG = 15

resids = pd.read_csv('residuals.csv')

# Base path to images
basePath = '../../MIT2kFaceDataset/2kfaces/'
imgnms = pd.read_csv('mostLeastAttractive.csv')

def main():
    # Get full paths to all images
    imgs = [basePath + str(k) for k in imgnms['locations']]

    # Plot PC1 vs PC2
    for k in range(1,6):
        x = resids['tier'].values.tolist()
        y = resids['residual'+str(k)].values.tolist()

        genPlot(x, y, imgs, k)


def genPlot(x, y, imgs, num):
    # Prepare the plot for putting thumbnails
    fig, ax = plt.subplots()
    fig.suptitle('Residual #' + str(num))

    # Add thumbnails
    imscatter(x, y, imgs, zoom=0.2, ax=ax)

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
        image = plt.imread(imgs[i%NUMIMG])
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