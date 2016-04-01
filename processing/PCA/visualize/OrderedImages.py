import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bokeh.charts import Bar, output_file, show
import pandas as pd
import matplotlib.pyplot as matplt
import scipy.stats as spstat
import bokeh.plotting as plt
import numpy as np
import sklearn.manifold as manifold
import ujson

def face2d_scatter_bokeh_plot(locs, img_list, html_file_name):
    p = plt.figure(plot_width=2400, plot_height=1200)
    p.image_url(x=locs[:, 0], y=locs[:, 1], url=img_list, w=3, h=3, anchor="center")
    p.circle(x=locs[:, 0], y=locs[:, 1])
    plt.output_file(html_file_name)
    plt.save(p)
    plt.show(p)
    return