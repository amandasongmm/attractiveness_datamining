"""
ScaleFeatures.py

The purpose of this script is to scale the different features to the same range
while maintaining variance across points

Built for Python2 in order to work best with machine learning libraries.

Author:   Chad Atalla
Date:     3/4/2016
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Load in data
features = pd.read_csv('allFeatures.csv')
features.drop(list(features.columns[x] for x in [0,1]), axis=1, inplace=True)

# Scale it
features = pd.DataFrame(preprocessing.scale(features, axis=0))

# Save it
features.to_csv('scaledFeatures.csv', index=False)