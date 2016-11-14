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
import numpy as np
p = pd.read_csv('./correlation_array', index_col = 0)
column_name = p.columns.tolist()
data = p.as_matrix()
print data.shape
data2 = pd.read_csv('./correlation_predicted.csv', index_col = 0)
data2 = data2.as_matrix()
print data2.shape

# plotHeatMap(rho, clusterNum=15, xTickLabel=social2Attr, \
#             colorMapName='coolwarm', figName='predictCorr_reordered1', fSize=3.5 \
#             , dendro=False)

index_order ,_ = plotHeatMap(data,clusterNum= 15,xTickLabel=column_name,\
            colorMapName='coolwarm',figName = 'correlation_dendro',fSize = 3.5\
            ,dendro=True)
print index_order

plotHeatMap(data2, clusterNum=15, xTickLabel=column_name, \
                             colorMapName='coolwarm', figName='predictCorr_reordered_dendro1', fSize=3.5 \
                             , dendro=True,tripIndex= index_order)

from scipy.signal import correlate2d
xcorr2d = correlate2d(data, data2, mode='same', boundary='fill', fillvalue=0)
#print xcorr2d

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def corr_coeff(A,B):
    #upper triangular.
    iu = np.triu_indices(A.shape[1],1)
    A_iu = A[iu]
    B_iu = B[iu]
    return np.corrcoef(A_iu,B_iu)

corr2d = corr_coeff(data,data2)
print corr2d[1,0]
