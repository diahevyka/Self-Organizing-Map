# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:43:10 2019

@author: Diah Hevyka M
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random

df = pd.read_csv("Dataset.csv", header=None)
data = []

def getRandom(min, max):
    rand = random.uniform(0, 1)
    return (rand * (max-min)) + min

for i in range(len(df)):
    data.append({'x' : df[0][i], 'y': df[1][i], 'color': "red"})


#Eucledian distance
def eucledian(x, y):
    return math.sqrt((x['x']-y['x'])**2 + (x['y']-y['y'])**2)

#Learningrate
lr = 0.1
tLr = 2

#neighborhood
sigma = 2
tSigma = 2

#Neuron
min_data_x = min(df[0])
min_data_y = min(df[1])
max_data_x = max(df[0])
max_data_y = max(df[1])
neuron = [
    {'x': getRandom(min_data_x, max_data_x), 'y': getRandom(min_data_y, max_data_y), 'color': 'blue', 'status': 'fixed'}, 
    {'x': getRandom(min_data_x, max_data_x), 'y': getRandom(min_data_y, max_data_y), 'color': 'black', 'status': 'fixed'},
    {'x': getRandom(min_data_x, max_data_x), 'y': getRandom(min_data_y, max_data_y), 'color': 'green', 'status': 'fixed'},
    {'x': getRandom(min_data_x, max_data_x), 'y': getRandom(min_data_y, max_data_y), 'color': 'yellow', 'status': 'fixed'},
    {'x': getRandom(min_data_x, max_data_x), 'y': getRandom(min_data_y, max_data_y), 'color': 'gray', 'status': 'fixed'},
    {'x': getRandom(min_data_x, max_data_x), 'y': getRandom(min_data_y, max_data_y), 'color': 'magenta', 'status': 'fixed'}       
]

# Plotdata awal
fig, ax = plt.subplots()
for i in data:
    ax.plot(i['x'], i['y'], ".", color=i['color'])
for i in neuron:
    ax.plot(i['x'], i['y'], "s", color=i['color'])

ax.set(xlabel='x', ylabel='y', title='Mapping Data Awal')
ax.grid()
plt.axis([2,19,2,19])
plt.show()

