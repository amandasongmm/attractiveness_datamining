#
# parseData.py
#
# The purpose of this script is to parse the twin attractiveness data
# of all image sets into a dataframe, sort, preprocess, and save to a csv.
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Author: 	Chad Atalla
# Date: 	1/23/2016
#



from parseSingleSet import parseSet
import pandas as pd 
import numpy as np
import re



'''
	Label:		ParseAllSets
	Purpose:	Parse the 4 different image sets
'''
# Beginning indexes of MIT Dataset
allData = parseSet(5,399)

# Beginning indexes of Glascow Dataset
allData = allData.append(parseSet(71, 465))

# Beginning indexes of Non-Glascow Dataset
allData = allData.append(parseSet(201, 595))

# Beginning indexes of 
allData = allData.append(parseSet(266, 660))



'''
	Label:		SaveResults
	Purpose:	Print out the dataframe and schema to file
'''
# Save the dataframe's schema and columns
allData.to_csv('../data/parsedData.csv')
