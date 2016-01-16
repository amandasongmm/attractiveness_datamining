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
import numpy as np

## TODO
## constants?
## Pull second twin data...

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

# Sort dataframe
data = data.sort_values(['image_id', 'twin_pair_id']);

# Print out the dataframe's schema and columns
print data.columns
print '\n'
print data