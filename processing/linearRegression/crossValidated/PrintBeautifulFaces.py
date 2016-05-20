"""
PrintBeautifulFaces.py

The purpose of this script is to identify the faces which most strongly correlate with the learned weights of beauty in faces

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   5/03/2016
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

faces = pd.read_csv('scaledFeatures.csv')
beauty = pd.read_csv('beautifulFeatures.csv')

# Beauty list
bList = [q[0] for q in beauty[[1]].values]

# Track results
pearsons = []

# Iterate over vectors and calculate correlation
for x in faces.values.tolist():
   pearsons.append(stats.pearsonr(x,bList)[0])

# Save lists
pearsonVals = pd.DataFrame(dict(face = range(200), pears = pearsons))
pearsonVals.sort_values(by='pears', inplace=True)
pearsonVals.to_csv('beautyValues.csv', index=False)

