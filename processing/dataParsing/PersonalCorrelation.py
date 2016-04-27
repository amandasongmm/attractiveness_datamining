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
# Remove bad indexes from Amanda
for x in [90, 93, 337, 417]:
    orig = orig[orig.pair_nums != x]
pair_nums = orig['pair_nums']
orig = orig.drop('pair_nums', 1)


# Load up repeats
repeat = pd.read_csv('RepeatVector.csv')
# Remove bad indexes from Amanda
for x in [90, 93, 337, 417]:
    repeat = repeat[repeat.pair_nums != x]
repeat = repeat.drop('pair_nums', 1)

# Track results
pearsons = []
spearmans = []

# Iterate over vectors and calculate correlation
for x,y in zip(orig.values.tolist(), repeat.values.tolist()):
   pearsons.append(stats.pearsonr(x,y)[0])
   spearmans.append(stats.spearmanr(x,y)[0])

# Save lists
pearsonVals = pd.DataFrame(dict(pair = pair_nums, pears = pearsons))
spearmanVals = pd.DataFrame(dict(pair = pair_nums, spears = spearmans))
pearsonVals.to_csv('pearsonVals.csv', index=False)
spearmanVals.to_csv('spearmanVals.csv', index=False)

# Calculate averages
print "Pearsonr Average: " + str(np.mean(pearsons))
print "Spearmanr Average: " + str(np.mean(spearmans))
