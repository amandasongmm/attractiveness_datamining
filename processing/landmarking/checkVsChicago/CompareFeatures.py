"""
CompareFeatures.py

This will calculate the differences between my features
and those from the Chicago Dataset

Author: Chad Atalla
Date:   4/6/2016
"""

import pandas as pd

# Pull in my data
mine = pd.read_csv('cfdFeatures.csv')

# Pull in the cfd data
cfd = pd.read_csv('actualFeatures.csv')

for x in range(200, 202):
    cur1 = (mine[mine.img_num == x]).values.tolist()[0]
    cur2 = (cfd[cfd.IMG == x]).values.tolist()[0]
    cur = zip(cur1, cur2)

    diff = [ abs(x-y)/(y) for x,y in cur ]
    print diff
    print ''
    print zip(([ x > .3 for x in diff ]), mine.columns)
    print ''
    print sum([ x > .3 for x in diff ])
    print ''
