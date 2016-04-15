"""
PersonalCorrelation.py

This script will compare the 2 vectors for each rater's repeated ratings

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/14/2016
"""

import pandas as pd 
import numpy as np
import scipy.stats as stats

# Load up originals
orig = pd.read_csv('OrigVector.csv')

# Load up repeats
repeat = pd.read_csv('RepeatVector.csv')

# Track results
pearsons = []
spearmans = []

# Iterate over vectors and calculate correlation
for x,y in zip(orig.values.tolist(), repeat.values.tolist()):
   pearsons.append(stats.pearsonr(x,y)[0])
   spearmans.append(stats.spearmanr(x,y)[0])

# Handle bad data
for x in (np.where((np.isnan(spearmans)) == True)[0].tolist()):
    del spearmans[x]
x = 0
for k in (np.where((np.isnan(pearsons)) == True)[0].tolist()):
    del pearsons[k-x]
    x += 1

# Calculate averages
print "Pearsonr Average: " + str(np.mean(pearsons))
print "Spearmanr Average: " + str(np.mean(spearmans))
