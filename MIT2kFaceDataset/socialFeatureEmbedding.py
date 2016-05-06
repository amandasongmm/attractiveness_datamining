import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

__author__ = 'amanda'

feature_array = pd.read_pickle('./clean_data/feature_array')

model = TSNE(n_components=2, random_state=0)
tsne_output = model.fit_transform(feature_array.T)

field_names = pd.read_csv('./clean_data/feature_field_list.txt')

fig, ax = plt.subplots()
ax.scatter( tsne_output[:, 0],  tsne_output[:, 1])


for i, txt in enumerate(field_names):
    ax.annotate(txt[:5], (tsne_output[i, 0], tsne_output[i, 1]), fontsize=15)
