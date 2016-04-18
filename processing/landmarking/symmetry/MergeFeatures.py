"""
MergeFeatures.py

The purpose of this script is to merge symmetry and averageness features
with the original geometric features.

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/12/2016
"""

import pandas as pd
import numpy as np

data = pd.read_csv('allOldFeatures.csv')
newData = pd.read_csv('AveragenessAndSymmetry.csv')

# Fit together new and old data
data = pd.concat([data, newData], axis=1)

data.to_csv('allFeatures.csv', index=False)