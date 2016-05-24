from plotFunc import plotHeatMap

import pandas as pd

p = pd.read_csv('./correlation_array', index_col = 0)
column_name = p.columns.tolist()
data = p.as_matrix()
#print data
plotHeatMap(data,clusterNum= 15,xTickLabel=column_name,\
            colorMapName='RdPu',figName = 'reorderedCorrelaionMatrix',fSize = 2\
            ,dendro = False)