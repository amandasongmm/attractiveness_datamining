"""
ExtractGenderAndAge.py

This script will extract gender and age for each twin

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   4/27/2016
"""

import pandas as pd

# The starts of the different twins' personal info
starts = [1,395]
vals = [starts[0] + y for y in range(4)] + [starts[1] + y for y in range(4)]

# Load up original data
origData = pd.read_csv('../../ratingData/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=vals)

# Ordering is ALL of twin1, then ALL of twin2
twin2 = pd.DataFrame(origData[range(4,8)])
twin1 = pd.DataFrame(origData[range(4)])

# Make col names uniform
twin1.columns = ['twinNum', 'gender', 'age', 'twinType']
twin2.columns = twin1.columns

# Concat data
twinData = twin1.append(twin2)

# Save dataframe
twinData.index.name = 'pair_nums'
twinData.to_csv('bioData.csv')
