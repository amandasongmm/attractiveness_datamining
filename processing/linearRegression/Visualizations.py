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



#
# Label:   PredictedVsActual
# Purpose: Plot predicted vs actual ratings
#
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



#
# Label:   PlotStatistics
# Purpose: Plot the MSE, Correlation and Variance Score
#
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



#
# Label:   OriginalWeights
# Purpose: Project PCs back to original features, and apply the weights
#           from linear regression to the original features, then plot
#
linRegWeights = pd.read_csv('linearCoefficients.csv')
PCCompositions = pd.read_csv('PCCompositions.csv', usecols=(range(6)))

# Normalize the weights of each PC (representing coverage of variance)
# before multiplying by the weight calculated by linear regression
for i in range (6):
    normalized = PCCompositions[str(i)]
    normalized = (normalized - normalized.mean()) / (normalized.max() - normalized.min())
    PCCompositions[str(i)] = normalized


# Calculate mean and deviation of linear regression weights of the PCs
linRegWeights = linRegWeights.as_matrix()
deviation = np.std(linRegWeights, axis=0)
linRegWeights = np.mean(linRegWeights, axis=0)

# The transformation coefficients from PCs to original features
PCCompositions = PCCompositions.as_matrix()

# Transform the PC weights to original 29 features
weights = np.multiply(PCCompositions, linRegWeights)
weights = np.mean(weights, axis=1)

# Transform the deviation from the PCs to the features
deviation = np.multiply(PCCompositions, deviation)
deviation = np.mean(deviation, axis=1)

# the x locations for the groups
x = range(1,30)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

rect = ax3.bar(x, weights, .15, yerr=deviation, error_kw={'ecolor':'Tomato', 'linewidth':2})

axes = plt.gca()

ax3.set_xticks(x)
ax3.set_xticklabels(range(1,30))
ax3.set_xlabel('Original Features')
ax3.set_ylabel('Weight from PCs')
plt.show()