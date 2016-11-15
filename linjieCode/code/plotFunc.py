import pylab
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
#from matplotlib import cm
# import scipy.spatial.distance as dist
# from scipy import stats
# import random
# import math
# import os

def plotHeatMap(data,xTickLabel = [],clusterNum = 20,colorMapName = 'RdBu',hLabel = '',title = '',savePath = './',figName = "",tripIndex = [],dendro = False,\
                bLabel = 'Correlation',lSize = 30,fSize = 1.5,\
                cBarTicks = [-0.6, -0.4, -0.2, 0, 0.2,0.4,0.6,0.8, 1.0]):
    # data is a length-by-length matrix
    # hLabel is the axis label, bLabel is the colorBar label
    # lSize is the label fontsize, fSize is the figure size
    # tripIndex is the index order, dendro is a boolean variable, True will show the dendrogram.
    length = data.shape[1]
    print 'Into plotting...'
    plt.interactive(True)
    fig = pylab.figure()
    Y = sch.linkage(data, method='centroid')
    if dendro:
        axdendro = fig.add_axes([0.05,0.1,0.2,0.8])
        axdendro.set_xticks([])
    Z = sch.dendrogram(Y, orientation='right',no_plot = not dendro)
    T = sch.fcluster(Y, clusterNum, 'maxclust')
    Tcount = T.tolist()
    Tcount = [Tcount.count(x) for x in Tcount]
    tempIndex = np.array(Tcount).argsort()[::-1]
    print tempIndex
    print Tcount
    print T
    print 'Finish dendrogram...'
    print 'Start plotting heatmap...'
    # Plot distance matrix.
    xIndex = range(1,length+1)
    if dendro:
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    else:
        axmatrix = fig.add_axes([0.2,0.05,0.64,0.75])
    index = Z['leaves']
#     print 'leaves',index
    if dendro or len(tripIndex) == 0:
        tripIndex = tempIndex
    data = data[tripIndex,:]
    data = data[:,tripIndex]
    im = axmatrix.matshow(data, aspect='auto',vmin=data.min(), vmax=data.max(), cmap = plt.get_cmap(colorMapName))
    if len(xTickLabel)>0:
        axmatrix.set_xticks(np.arange(0,length)[::2], minor=False)
        #xTickLabel = [ xTickLabel[i] for i in tripIndex]
        axmatrix.set_xticklabels(xTickLabel[::2],fontsize = lSize-10,rotation=90)
        axmatrix.set_yticks(np.arange(0, length)[::2]+1, minor=False)
        axmatrix.set_yticklabels(xTickLabel[1:][::2], fontsize=lSize - 10)
    else:
        axmatrix.set_yticks([])
        axmatrix.set_xticks([])
    pylab.xlabel(r'\textbf{'+hLabel+'}',fontsize = lSize)


    # Plot colorbar.
    axcolor = fig.add_axes([0.85,0.05,0.015,0.75])
    # adjusting your colorbar limit
    cbar = pylab.colorbar(im, cax=axcolor,ticks=cBarTicks)
    cbar.set_label(bLabel, rotation=270,labelpad=30,fontsize = lSize)
    cbar.ax.tick_params(labelsize=lSize)
    # Display and save figure.
    showFig = fig.show()
    fig = pylab.gcf()
    fig.canvas.set_window_title(title)
    fig.set_size_inches(5*fSize, 5*fSize)
    if len(figName)>0:
        fig.savefig(savePath+figName+'.png')
    return tempIndex,T

