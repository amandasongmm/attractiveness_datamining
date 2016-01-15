#
# parseData.py
#
# The purpose of this script is to parse the twin attractiveness data
# into a pandas dataframe where it can be sorted and processed.
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Authors: 	Amanda Song, Chad Atalla
# Date: 	1/15/2016
#

import pandas as pd 

# Import data from csv into pandas dataframe
data = pd.read_csv('../germine_CurrentBiology_twinaesthetics_PUBLIC.csv')

# Print out the dataframe's schema and columns
print data.columns
print data