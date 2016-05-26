import pandas as pd
import simplejson

__author__ = 'amanda'

'''
This code extracts 40 social features from the psychology-attributes.xlsx,
saves three variables: feature_array, img_name_list, feature_fields in the clean_data folder.
'''

file_name = '../Full Attribute Scores/psychology attributes/psychology-attributes.xlsx'
xl_file = pd.ExcelFile(file_name)
dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
df = dfs['Final Values']  # only use the group average data

# save image file name list
img_name_list = df['Filename']
img_name_list.to_pickle('../clean_data/img_name_list')

# Delete irrelevant information.
delete_field_list = ['Filename', 'Image #', 'catch', 'catchAns', 'subage', 'submale', 'subrace',
                     'catch.1', 'catchAns.1', 'subage.1', 'submale.1', 'subrace.1']

# save feature array
df = df.drop(delete_field_list, axis=1, inplace=True)
df.to_pickle('../clean_data/feature_array')  # How to load: data = pd.read_pickle('./clean_data/feature_array')


# Obtain and save feature field list.
feature_fields = list(df.columns.values)   # Get the field names of the remaining 40 social features.
f = open('../clean_data/feature_field_list.txt', 'w')
simplejson.dump(feature_fields, f)
f.close()

'''
feature_array and img_name_list are in pickle format,  use pd.read_pickle(file_path) to load the data.
feature_fields is in txt format.
'''

