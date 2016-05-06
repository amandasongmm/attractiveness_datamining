import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

__author__ = 'amanda'

feature_array = pd.read_pickle('../clean_data/feature_array')

model = TSNE(n_components=2, random_state=0)
tsne_output = model.fit_transform(feature_array.T)

np.savez('../clean_data/tsne_social_feature_embedding', tsne_output=tsne_output)
