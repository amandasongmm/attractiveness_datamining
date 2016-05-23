import pylab
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
# import matplotlib.pyplot as plt
# import scipy.spatial.distance as dist
# from scipy import stats
# import random
# import math
# import os

def plotHeatMap(data,title = '',savePath = '',figName = "",tripIndex = [],dendro = False,hLabel = 24,\
                bLabel = 24,lSize = 30,fSize = 1.5, cBarTicks = [-0.6, -0.4, -0.2, 0, 0.2,0.4,0.6,0.8, 1.0]):
    # data is a length-by-length matrix
    # hLabel is the axis label, bLabel is the colorBar label
    # lSize is the label fontsize, fSize is the figure size
    # tripIndex is the index order, dendro is a boolean variable, True will show the dendrogram.
    length = data.shape[1]
    print 'Into plotting...'
    fig = pylab.figure()
    Y = sch.linkage(data, method='centroid')
    if dendro:
        axdendro = fig.add_axes([0.05,0.1,0.2,0.8])
        axdendro.set_xticks([])
    Z = sch.dendrogram(Y, orientation='right',no_plot = not dendro)
    T = sch.fcluster(Y, 20, 'maxclust')
    Tcount = T.tolist()
    Tcount = [Tcount.count(x) for x in Tcount]
    tempIndex = np.array(Tcount).argsort()[::-1]
#     print tempIndex
    print Tcount
    print T
    print 'Finish dendrogram...'
    print 'Start plotting heatmap...'
    # Plot distance matrix.
    xIndex = range(1,length+1)
    if dendro:
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    else:
        axmatrix = fig.add_axes([0.1,0.1,0.7,0.8])
    index = Z['leaves']
#     print 'leaves',index
    if dendro or len(tripIndex) == 0:
        tripIndex = tempIndex
    data = data[tripIndex,:]
    data = data[:,tripIndex]
    im = axmatrix.matshow(data, aspect='auto', origin='lower',vmin=data.min(), vmax=data.max())
    axmatrix.set_xticks(np.arange(0,length,200), minor=False)
    axmatrix.set_xticklabels(tripIndex[::200],fontsize = lSize-10)
    pylab.xlabel(r'\textbf{'+hLabel+'}',fontsize = lSize)
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.81,0.1,0.02,0.8])
    # adjusting your colorbar limit
    cbar = pylab.colorbar(im, cax=axcolor,ticks=cBarTicks)
    cbar.set_label(bLabel, rotation=270,labelpad=30,fontsize = lSize)
    cbar.ax.tick_params(labelsize=lSize)
    # Display and save figure.
    fig.show()
    fig = pylab.gcf()
    fig.canvas.set_window_title(title)
    fig.set_size_inches(5*fSize, 5*fSize)
    if len(figName)>0:
        fig.savefig(savePath+figName+'.png')
    return tempIndex,T