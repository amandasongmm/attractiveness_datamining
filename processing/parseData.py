#
# parseData.py
#
# The purpose of this script is to parse the twin attractiveness data
# into a pandas dataframe, sort, preprocess, and save to a csv.
#
# Built for Python2 in order to work best with machine learning libraries.
#
# Authors: 	Amanda Song, Chad Atalla
# Date: 	1/15/2016
#

import pandas as pd 
import numpy as np



'''
	Label:		ParseTwin1Data
	Purpose: 	Parse the data for twin 1 into a dataframe
'''
# Import first twins' MIT data from csv into pandas dataframe
data = pd.read_csv('../data/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=range(0,70))

# Rename columns in dataframe to match our schema
data.rename(columns={'Unnamed: 0': 'twin_pair_id', 'Twin_Num_of2.twin1': 'twin_id', 'sex_x.twin1' : 'gender', 'age.twin1' : 'age', 'Zygosity.twin1' : 'twin_type'}, inplace=True)

# Melt the data to expand the MIT face columns onto multiple rows
data = pd.melt(data, id_vars=['twin_pair_id', 'twin_id', 'gender', 'age', 'twin_type'], var_name='image_id', value_name='rating')

# Fix numbering of images, MIT set will be 1 - 65
data['image_id'] = data['image_id'].replace(['Faces.MIT', '.twin1'], ['',''], regex=True)
data['image_id'] = data['image_id'].astype(int)

# Re-order columns to match schema
cols = ['image_id', 'twin_pair_id', 'twin_type', 'twin_id', 'age', 'gender', 'rating']
data = data.ix[:, cols]



'''
	Label:		ParseTwin2Data
	Purpose: 	Parse the data for twin 2 into a dataframe
'''
# Import second twins' MIT data from csv into pandas dataframe
# Grab twin pair id from front, twin2 data from middle of csv
data2 = pd.read_csv('../data/germine_CurrentBiology_twinaesthetics_PUBLIC.csv', usecols=(range(1) + range(395, 464)))

# Rename columns in dataframe to match our schema
data2.rename(columns={'Unnamed: 0': 'twin_pair_id', 'Twin_Num_of2.twin2': 'twin_id', 'sex_x.twin2' : 'gender', 'age.twin2' : 'age', 'Zygosity.twin2' : 'twin_type'}, inplace=True)

# Melt the data to expand the MIT face columns onto multiple rows
data2 = pd.melt(data2, id_vars=['twin_pair_id', 'twin_id', 'gender', 'age', 'twin_type'], var_name='image_id', value_name='rating')

# Fix numbering of images, MIT set will be 1 - 65
data2['image_id'] = data2['image_id'].replace(['Faces.MIT', '.twin2'], ['',''], regex=True)
data2['image_id'] = data2['image_id'].astype(int)

# Re-order columns to match schema
data2 = data2.ix[:, cols]



'''
	Label:		SortValues
	Purpose:	Combine and sort the data from the two twins
'''
# Combine twin1 and twin2 data
allData = data.append(data2)

# Sort dataframe
allData = allData.sort_values(['image_id', 'twin_pair_id', 'twin_id']);



'''
	Label:		SaveResults
	Purpose:	Print out the dataframe and schema to file
'''
# Save the dataframe's schema and columns
allData.to_csv('../data/parsedData.csv')
