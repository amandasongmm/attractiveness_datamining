'''
Use this code for iPython only
import sys
print sys.path
# local
PkgPath = '/Users/Olivialinlin/Documents/Github/attractiveness_datamining/linjieCode/code'
# server
# PkgPath = '/home/lli-ms/attractiveness_datamining/linjieCode/code'

if PkgPath not in sys.path:
    sys.path.insert(0, PkgPath)
'''
from plotFunc import plotHeatMap
import pandas as pd
p = pd.read_csv('./correlation_array', index_col = 0)
column_name = p.columns.tolist()
data = p.as_matrix()
#print data
plotHeatMap(data,clusterNum= 15,xTickLabel=column_name,\
            colorMapName='coolwarm',figName = 'reorderedCorrelaionMatrix',fSize = 3.5\
            ,dendro = False)