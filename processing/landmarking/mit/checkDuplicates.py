import pandas as pd

data = pd.read_csv('combinedLandmarks.csv')

for k in range(2207 * 69 - 1):
	x = data.iloc[k]['point']
	y = data.iloc[k+1]['point']
	if x == y:
		print data.iloc[k+1]['image']
