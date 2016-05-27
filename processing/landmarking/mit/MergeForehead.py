import pandas as pd 

data = pd.read_csv('mitLandmarks.csv')
fore = pd.read_csv('mitLandmarksForehead.csv')

data = data.append(fore)
data = data.sort_values(by=['image', 'point'])

data.to_csv('combinedLandmarks.csv', index=False)