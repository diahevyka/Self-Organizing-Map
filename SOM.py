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
cluster = []
color = ['gray', 'yellow', 'green', 'fuchsia', 'hotpink', 'maroon', 'steelblue', 'aqua', 'lime', 'peru']

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

#Normalisasi data
def normalisasi(data):
    nmin = {'x': min([n['x'] for n in data]), 'y': min([n['y'] for n in data])}
    nmax = {'x': max([n['x'] for n in data]), 'y': max([n['y'] for n in data])}
    for i in data:
        i['x'] = (i['x'] - nmin['x']) / (nmax['x'] - nmin['x']) 
        i['y'] = (i['y'] - nmin['y']) / (nmax['y'] - nmin['y']) 
    return data

normal = normalisasi(data)

#Neuron
neuron = []
neuron_size = 1200
for i in range(neuron_size):
    neuron.append({'x': random.uniform(0, 1), 'y': random.uniform(0, 1), 'color': 'blue', 'status': 'fixed'})

fig, ax = plt.subplots()
for i in neuron:
    ax.plot(i['x'], i['y'], ".", color=i['color'])
for i in normal:
    ax.plot(i['x'], i['y'], ".", color=i['color'])


ax.set(xlabel='x', ylabel='y', title='Mapping Data Awal')
ax.grid()
plt.axis([-0.1,1.1,-0.1,1.1])
plt.show()

iterasi = 100
konvergen = 0.00000001
for t in range(iterasi):
    #inisialisasi input secara random sebagai neighboor
    rand = random.randint(1, len(data)-1)
    x = data[rand]
    
    #mencari win neuron
    win_neuron = neuron[0]
    for a in neuron:
        a['status'] = 'neuron'
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
        print("iterasi: ",t)
        break
    
    #jika tidak konvergen, tambah cluster
    if (win_neuron not in cluster): cluster.append(win_neuron)
    
    #kalau tidak konvergen, update lr dan sigma
    lr *= np.exp(-t/tLr)
    tSigma *= np.exp(-t/tSigma)


#menentukan warna cluster
for i in range(len(cluster)):
    cluster[i]['color'] = color[i]
    

#Menentukan cluster dataset
for d in normal:
    win_cluster = cluster[0]
    for klaster in cluster:
        if eucledian(d, klaster) < eucledian(d, win_cluster): win_cluster = klaster
    d['color'] = win_cluster['color']
    #if (d['color'] not in cluster): cluster.append(d['color'])
    

#print('cluster = ', len(cluster))

#Plot data akhir

print("Jumlah cluster: ", len(cluster))

fig, ax = plt.subplots()
for i in normal:
    ax.plot(i['x'], i['y'], ".", color=i['color'])


ax.set(xlabel='x', ylabel='y', title='Mapping Data Akhir')
ax.grid()
plt.axis([-0.1,1.1,-0.1,1.1])
plt.show()
