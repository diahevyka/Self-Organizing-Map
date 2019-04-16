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

#
iterasi = 5
konvergen = 0.000001
for t in range(iterasi):
    #inisialisasi input secara random sebagai neighboor
    rand = random.randint(1, len(data)-1)
    x = data[rand]
    
    #mencari win neuron
    win_neuron = neuron[0]
    for a in neuron:
        a['status'] = 'fixed'
        if eucledian(x, a) < eucledian(x, win_neuron): win_neuron = a
    
    #mencari tetangga win neuron
    for b in neuron:
        if eucledian(b, win_neuron) < sigma:
            b['status'] = 'neighborhood'
    
    #update weigh untuk set
    for c in neuron:
        if (c['status'] == 'neighborhood'):
            s = eucledian(win_neuron, c)
            phi = np.exp(-(s**2 / (2*sigma**2)))
            dWeight = lr * phi * eucledian(x, c)
            c['x'] += dWeight
            c['y'] += dWeight
            
    #cek konvergen
    if (dWeight < konvergen):
        break
    
    #kalau tidak konvergen, update lr dan sigma
    lr *= np.exp(-t/tLr)
    tSigma *= np.exp(-t/tSigma)
    
#Menentukan cluster dataset
cluster = []
for d in data:
    win_neuron = neuron[0]
    for e in neuron:
        if eucledian(d, e) < eucledian(d, win_neuron): win_neuron = e
    d['color'] = win_neuron['color']
    if (d['color'] not in cluster): cluster.append(d['color'])
    

print('cluster = ', len(cluster))

#Plot data akhir
fig, ax = plt.subplots()
for i in data:
    ax.plot(i['x'], i['y'], ".", color=i['color'])
for i in neuron:
    ax.plot(i['x'], i['y'], "s", color=i['color'])

ax.set(xlabel='x', ylabel='y', title='Mapping Data Akhir')
ax.grid()
plt.axis([2,19,2,19])
plt.show()