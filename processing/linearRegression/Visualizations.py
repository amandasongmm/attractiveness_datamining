"""
Visualization.py

The purpose of this script is to visualize the results from linear regression

Built for Python2 in order to work best with machine learning libraries.

Author: Chad Atalla
Date:   2/19/2016
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Plot predicted vs actual
predicted = pd.read_csv('predictedRatings.csv')
original = pd.read_csv('ratingMatrixChad.csv')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

for i in range(predicted.index.size):
    ax1.scatter(predicted.loc[i],original.loc[i])

plt.xlabel('Predicted rating')
plt.ylabel('Actual rating')
fig1.suptitle('Predicted vs Actual Ratings')
plt.show()



# Create plot for MSE, Correlation, and Variance Score
correlation = pd.read_csv('correlations.csv')
variance = pd.read_csv('varianceScore.csv')
mse = pd.read_csv('mseVals.csv')

correlation = correlation.as_matrix()
variance = variance.as_matrix()
mse = mse.as_matrix()

results = np.array((np.mean(correlation), np.mean(variance), np.mean(mse)))
resultsDev = np.array((np.std(correlation), np.std(variance), np.std(mse)))

# the x locations for the groups
x = range(1,4)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

rect = ax2.bar(x, results, .35, yerr=resultsDev, error_kw={'ecolor':'Tomato', 'linewidth':2})

axes = plt.gca()
axes.set_ylim([0,3])

ax2.set_xticks([1,2,3])
ax2.set_xticklabels(('Correlation','VarianceScore','MeanSquaredError'))
plt.show()


# Plot weights of PCs projected back to original features
coefficients = pd.read_csv('PCCompositions.csv')
for i in range (29):
    normalized = coefficients[str(i)]
    normalized = (normalized - normalized.mean()) / (normalized.max() - normalized.min())
    coefficients[str(i)] = normalized

weights = coefficients.as_matrix()
#weightDev = np.std(weights, axis=0)
weights = np.mean(weights, axis=1)

# the x locations for the groups
x = range(1,30)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

rect = ax3.bar(x, weights, .15) #, yerr=weightDev, error_kw={'ecolor':'Tomato', 'linewidth':2})

axes = plt.gca()

ax3.set_xticks(x)
ax3.set_xticklabels(range(1,30))
ax3.set_xlabel('Original Features')
ax3.set_ylabel('Weight from PCs')
plt.show()