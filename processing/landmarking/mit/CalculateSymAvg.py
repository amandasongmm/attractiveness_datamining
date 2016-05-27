"""
CalculateSymAvg.py

The purpose of this script is to run calculate the new features of symmetry
and averageness according to the
Facial Attractiveness: Symmetry and Averageness paper.


Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/12/2016
"""

NUM_IMG = 2207
NUM_LANDMARK = 68

import pandas as pd
import numpy as np
from scipy.spatial import distance

# Pull in the raw landmarks
landmarks = pd.read_csv('mitLandmarksT.csv')

# This dataframe will store the end-features
results = pd.DataFrame(columns=('imgName', 'FA', 'CA', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'lh', 'AV'))

# Loop over all landmarks for each of the 200 images
for i in range(NUM_IMG):
    # Get current file name
    curName = (landmarks.loc[i*NUM_LANDMARK])['image']

    # Create a list where all landmarks can be temporarily stored for this img
    p = []

    # Loop over each landmark for this image
    for j in range(NUM_LANDMARK):
        # Add the current points to the curLandmarks list
        row = landmarks[landmarks.image == curName][landmarks.point == j]
        p.append((int(row['x']), int(row['y'])))

    #
    # Now extract new features from this image's landmarks
    #

    # First, calculate symmetry,
    #   m1 - m6 are horizontal lines across facial features.
    m1 = tuple((x-y)/2 + y for x,y in zip(p[36], p[45]))
    m2 = tuple((x-y)/2 + y for x,y in zip(p[39], p[42]))
    m3 = tuple((x-y)/2 + y for x,y in zip(p[0], p[16]))
    m4 = tuple((x-y)/2 + y for x,y in zip(p[31], p[35]))
    m5 = tuple((x-y)/2 + y for x,y in zip(p[4], p[12]))
    m6 = tuple((x-y)/2 + y for x,y in zip(p[48], p[54]))

    # The two end-result asymmetry features
    FA = sum([distance.euclidean(m1 , m2), distance.euclidean(m1 , m3), distance.euclidean(m1 , m4), distance.euclidean(m1 , m5), distance.euclidean(m1 , m6), distance.euclidean(m2 , m3), distance.euclidean(m2 , m4), distance.euclidean(m2 , m5), distance.euclidean(m2 , m6), distance.euclidean(m3 , m4), distance.euclidean(m3 , m5), distance.euclidean(m3 , m6), distance.euclidean(m4 , m5), distance.euclidean(m4 , m6), distance.euclidean(m5 , m6)])

    CFA = sum([distance.euclidean(m1 , m2), distance.euclidean(m2 , m3), distance.euclidean(m3 , m4), distance.euclidean(m4 , m5), distance.euclidean(m5 , m6)])


    # Now, calculate the features needed for averageness
    l1 = distance.euclidean(p[36], p[45])
    l2 = distance.euclidean(p[39], p[42])
    l3 = distance.euclidean(p[0], p[16])
    l4 = distance.euclidean(p[31], p[35])
    l5 = distance.euclidean(p[4], p[12])
    l6 = distance.euclidean(p[48], p[54])
    lh = distance.euclidean(p[8], m1) + distance.euclidean(p[8], m5)

    oneLine = pd.Series([curName, FA, CFA, l1, l2, l3, l4, l5, l6, lh, 0], index=results.columns)
    results.loc[i] = oneLine


# Results has been populated with data, still need to calculate averageness
avg1 = np.mean(results['l1'])
avg2 = np.mean(results['l2'])
avg3 = np.mean(results['l3'])
avg4 = np.mean(results['l4'])
avg5 = np.mean(results['l5'])
avg6 = np.mean(results['l6'])
avgh = np.mean(results['lh'])

# Calculate personal averageness
for x in range(NUM_IMG):
    averageness = sum([abs(avg1 - results.loc[x]['l1']), abs(avg2 - results.loc[x]['l2']), abs(avg3 - results.loc[x]['l3']), abs(avg4 - results.loc[x]['l4']), abs(avg5 - results.loc[x]['l5']), abs(avg6 - results.loc[x]['l6']), abs(avgh - results.loc[x]['lh'])])
    results.set_value(x,'AV', averageness)

results[['imgName', 'FA', 'CA', 'AV']].to_csv('mitAveragenessAndSymmetry.csv', index=False)