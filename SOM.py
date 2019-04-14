# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:43:10 2019

@author: Diah Hevyka M
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Dataset.csv", header=None)

# Data for plotting
data1 = [1.2, 2.2 , 3.1, 3.3, 4.2]

fig, ax = plt.subplots()
ax.plot([1,2],[2,4],"ro", color="blue")

ax.set(xlabel='x', ylabel='y',
       title='Mapping Data')
ax.grid()

fig.savefig("test.png")
plt.show()
