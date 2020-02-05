#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:26:59 2020

@author: sebas
"""
# Mapping cell engulfment for different TS cell number and (d,R) pairs.
# Key of values:
# P1 -> Iteration (dummy)
# P2 -> TS cells center displacement (d)
# P3 -> TS cells initial radius (R)
# P4 -> TS cell popullation (pop_TS)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os



root_path = '/Users/sebas/morpheus/'
# Only input should be to tell path of folder where the sweep was performed
path = '/Users/sebas/morpheus/Example-CellSorting-2D_sweep_38/'

sweeps = os.listdir(path)
sweeps.remove('sweep_header.csv')
sweeps.remove('sweep_data.csv')
#sweeps.remove('.DS_Store')
sweeps.sort()

# Save number of iterations as the P1 in Morpheus
sweep_data = pd.read_csv(path + 'sweep_data.csv',sep='\t',header=(0))
iterations = sweep_data['P1'].iloc[-1] + 1 #+1 depends whether we start in 0 or 1
disp = sweep_data['P2'].unique()
radii = sweep_data['P3'].unique()
ts_pop = sweep_data['P4'].unique()

# We will store the data for each TS population number separately:
data1 = np.zeros(iterations*len(radii)*len(disp))
data2 = np.zeros(iterations*len(radii)*len(disp))

for i in range(len(sweeps)):
    sweep_path = path + sweeps[i] + '/'
    CT1_bound = pd.read_csv(sweep_path + 'logger.csv',sep='\t',header=(0))
    if i%len(ts_pop) == 0:
        data1[int(i/2)] = CT1_bound['boundaryCT1Med'].iloc[-1]
    else:
        data2[int((i-1)/2)] = CT1_bound['boundaryCT1Med'].iloc[-1]

# Reshape data according to number of iterations, d and R. We have 6 pair of values
        #in total, how to group them and organize them??
# Again, first array is the iteration, row refers to 'd' and column to 'R'
data_1 = np.reshape(data1,(iterations,len(disp),len(radii)))
data_2 = np.reshape(data2,(iterations,len(disp),len(radii)))


## Not sure about this but let's see

# Now we take the probability of engulfment for 'data':
data_1_prob = np.zeros((len(disp),len(radii)))
data_2_prob = np.zeros((len(disp),len(radii)))


# Fix one specific value of radius,alfa and iterate over all iterations:
for i in range(len(disp)):
    for j in range(len(radii)):
        count = 0
        for k in range(iterations):
            if data_1[k,i,j] == 0:
                count += 1
            else:
                count += 0
        data_1_prob[i,j] = count*100/iterations

for i in range(len(disp)):
    for j in range(len(radii)):
        count = 0
        for k in range(iterations):
            if data_2[k,i,j] == 0:
                count += 1
            else:
                count += 0
        data_2_prob[i,j] = count*100/iterations




x = np.array([ item for item in disp for r in range(len(radii)) ])
y = np.array([radii for j in range(len(disp))]).flatten()
n1 = data_1_prob.flatten() #Probabilities
n2 = data_2_prob.flatten() #Probabilities

fig, ax = plt.subplots()
ax.plot(x, y,'r+')
plt.xlabel('d')
plt.ylabel('R')
plt.title('Probability of engulfment for ' + str(iterations) + ' iterations and TS = '+str(ts_pop[0]))

for i, txt in enumerate(n1):
    ax.annotate(int(txt), (x[i],y[i]))

plt.grid()
plt.savefig('/Users/sebas/morpheus/Results/Prob_Den for '+str(iterations)+' iterations and TS = '+str(ts_pop[0])+'.png')

fig, ax = plt.subplots()
ax.plot(x, y,'r+')
plt.xlabel('d')
plt.ylabel('R')
plt.title('Probability of engulfment for ' + str(iterations) + ' iterations and TS = '+str(ts_pop[1]))

for i, txt in enumerate(n2):
    ax.annotate(int(txt), (x[i],y[i]))

plt.grid()
plt.savefig('/Users/sebas/morpheus/Results/Prob_Den for '+str(iterations)+' iterations and TS = '+str(ts_pop[1])+'.png')





















