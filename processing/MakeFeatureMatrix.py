"""
MakeFeatureMatrix.py

The purpose of this script is to create a matrix of all raters (sorted by
twin pair ID and then twin # 1 or 2) and their ratings for 200 faces

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/18/2016
"""


import pandas as pd


origData = pd.read_csv('../ratingData/parsedData.csv')

# Destination blank dataframe
newData = pd.DataFrame(columns=range(1,201))

# Iterate over all twins
for i in range(1,1549):
    twinNum = ((i-1)%2) + 1
    pairNum = int((i+1)/2)

    # Skip the bad entries
    if pairNum != 94 and pairNum != 418 and pairNum != 338:
        # Grab the next person's ratings and add them to dataframe
        oneRow = ((origData[origData.twin_pair_id == pairNum][origData.twin_id == twinNum])['rating']).map(lambda x: float(x)).as_matrix()

        newData.loc[i] = oneRow

newData.to_csv('ratingMatrixChad.csv', index=False)