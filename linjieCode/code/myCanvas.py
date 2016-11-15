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
p = pd.read_csv('../correlation_heatmap/correlation_array', index_col = 0)
column_name = p.columns.tolist()
data = p.as_matrix()
print data.shape
data2 = pd.read_csv('../correlation_heatmap/correlation_predicted.csv', index_col = 0)
data2 = data2.as_matrix()
print data2.shape

# plotHeatMap(rho, clusterNum=15, xTickLabel=social2Attr, \
#             colorMapName='coolwarm', figName='predictCorr_reordered1', fSize=3.5 \
#             , dendro=False)
index_order = [21, 27, 26, 34, 11, 22, 2, 13, 12, 5, 28, 23, 33, 9, 29, 24, 32, 35, 25, 4, 14, 18, 17, 10, 31, 3, 20, 39, 30, 36, 6, 7, 19, 38, 1, 15, 8, 0, 16, 37]
tickLable = [column_name[temp] for temp in index_order]#column_name[::2]
plotHeatMap(data,clusterNum= 1,xTickLabel=tickLable,\
            colorMapName='coolwarm',figName = 'correlation_dendro',fSize = 2\
            ,dendro=False,tripIndex= index_order)
print index_order
#tickLable = column_name[1:][::2]
plotHeatMap(data2, clusterNum=10, xTickLabel=tickLable, \
                             colorMapName='coolwarm', figName='predictCorr_reordered_dendro1', fSize=2\
                             , dendro=False,tripIndex= index_order)

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
