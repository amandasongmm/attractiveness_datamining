import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_ELEM = 32

column_labels = list([ 'f' + str(x) for x in range(NUM_ELEM)])
row_labels = list([ 'p' + str(x) for x in range(NUM_ELEM)])
data = pd.read_csv('PCCompositions.csv')
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.seismic)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)
plt.show()