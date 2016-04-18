"""
ExtractPersonalVectors.py

This script will extract 2 vectors of length 60 for each rater, including
all of their duplicated ratings (15 for each dataset) x 2 times

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/14/2016
"""

import pandas as pd

# The starts of the different datasets for each twin
starts = [5,399,70,464,200,594,265,659]

# The positions of the repeat states
repeats = [z for y in (range(x+50, x+65) for x in starts) for z in y]
repeats.sort()

# The positions of the original data for repeated states
origNums = [3 * x for x in range(1,14)] + [43, 47]
origs = [x + y - 1 for y in origNums for x in starts] 
origs.sort()

# Load up original data
origData = pd.read_csv('../../ratingData/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=origs)

# Load up repeat data
repeatData = pd.read_csv('../../ratingData/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=repeats)

# Ordering is ALL of twin1, then ALL of twin2
twin2 = pd.DataFrame(origData[range(60,120)])
twin1 = pd.DataFrame(origData[range(60)])

# Make col names uniform
twin1.columns = [str(x) for x in range(60)]
twin2.columns = twin1.columns

# Concat data
origTwins = twin1.append(twin2)

# Save dataframes
origTwins.to_csv('OrigVector.csv', index=False)

# Ordering is ALL of twin1, then ALL of twin2
twin2 = pd.DataFrame(repeatData[range(60,120)])
twin1 = pd.DataFrame(repeatData[range(60)])

# Make col names uniform
twin1.columns = [str(x) for x in range(60)]
twin2.columns = twin1.columns

# Concat data
repeatTwins = twin1.append(twin2)

# Save dataframes
repeatTwins.to_csv('RepeatVector.csv', index=False)
