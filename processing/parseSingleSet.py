#
# parseSingleSet.py
#
# The purpose of this function is to parse the ratings from both twins for a
# single set of images (MIT, Glascow, etc.)
#
# Author: 	Chad Atalla
# Date: 	1/23/2016
#



import pandas as pd 
import numpy as np
import re



# Define constants
TWIN_1_INFO = range(5)					 # Range of personal info for twin1
TWIN_2_INFO = range(1) + range(395, 399) # Range of personal info for twin2

AMT_PER_SET = 50	# Number of non-repeating images from each set



def parseSet (twin1Start, twin2Start):
	'''
		Label:		ParseTwinData
		Purpose: 	Parse the data for twin 1 and 2 into a dataframe
	'''
	# Import first twins' MIT data from csv into pandas dataframe
	data = pd.read_csv('../data/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=(TWIN_1_INFO + range(twin1Start, twin1Start+AMT_PER_SET)))

	# Rename columns in dataframe to match our schema
	data.rename(columns={'Unnamed: 0': 'twin_pair_id', 'Twin_Num_of2.twin1': 'twin_id', 'sex_x.twin1' : 'gender', 'age.twin1' : 'age', 'Zygosity.twin1' : 'twin_type'}, inplace=True)

	# Import second twins' MIT data from csv into pandas dataframe
	# Grab twin pair id from front, twin2 data from middle of csv
	data2 = pd.read_csv('../data/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=(TWIN_2_INFO + range(twin2Start, twin2Start+AMT_PER_SET)))

	# Rename columns in dataframe to match our schema
	data2.rename(columns={'Unnamed: 0': 'twin_pair_id', 'Twin_Num_of2.twin2': 'twin_id', 'sex_x.twin2' : 'gender', 'age.twin2' : 'age', 'Zygosity.twin2' : 'twin_type'}, inplace=True)



	'''
		Label:		CombineData
		Purpose: 	Melt and combine the data for twin 1 and 2
	'''
	# Melt the data to expand the MIT face columns onto multiple rows
	data = pd.melt(data, id_vars=['twin_pair_id', 'twin_id', 'gender', 'age', 'twin_type'], var_name='image_id', value_name='rating')

	# Melt the data to expand the MIT face columns onto multiple rows
	data2 = pd.melt(data2, id_vars=['twin_pair_id', 'twin_id', 'gender', 'age', 'twin_type'], var_name='image_id', value_name='rating')

	# Append data
	data = data.append(data2)



	'''
		Label:		ProcessData
		Purpose:	Map the data to our desired schema
	'''
	# Create column for indicating picture dataset (Strip face and twin info)
	data['dataset'] = data['image_id'].map(lambda x: x[6: (re.search('\d', x)).start()])

	# Fix numbering of images
	data['image_id'] = data['image_id'].map(lambda x: x[re.search('\d', x).start() : x.find('.', 7)])
	data['image_id'] = data['image_id'].astype(int)

	# Re-order columns to match schema
	cols = ['dataset', 'image_id', 'twin_pair_id', 'twin_type', 'twin_id', 'age', 'gender', 'rating']
	data = data.ix[:, cols]

	# Sort dataframe
	data = data.sort_values(['dataset', 'image_id', 'twin_pair_id', 'twin_id'])

	return data