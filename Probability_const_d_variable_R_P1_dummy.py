#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 05:46:55 2019

@author: sebas
"""
######### Important: change to morpheus folder but not necessary?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
# Function for analysing probability over a line.
# Idea is to create some sort of surface plot showing the probabilities of 
# engulfment for a fixed distance from center and ratio of total grid.


root_path = '/Users/sebas/morpheus/'
# Only input should be to tell path of folder where the sweep was performed
path = '/Users/sebas/morpheus/Example-CellSorting-2D_sweep_36/'
# Working for sweep_30
### WICHTIG, it is important the order of the Parameter sweep: this program
#works with P1 as the dummy variable.

d_cen = 0

sweeps = os.listdir(path)
sweeps.remove('sweep_header.csv')
sweeps.remove('sweep_data.csv')
#sweeps.remove('.DS_Store')
sweeps.sort() # Get all folders sorted

# Save number of iterations:
sweep_data = pd.read_csv(path + 'sweep_data.csv',sep='\t',header=(0))
iterations = sweep_data['P1'].iloc[-1] + 1 # +1 comes from iterations zero
# Save array with scanned distances:
distances = sweep_data['P2'].unique()
# Create zero array to store simulated data in 1D:
data = np.zeros(len(distances)*iterations)


for i in range(len(sweeps)):
    sweep_path = path + sweeps[i] + '/'
    CT1_bound = pd.read_csv(sweep_path + 'logger.csv',sep='\t',header=(0))
    data[i] = CT1_bound['boundaryCT1Med'].iloc[-1]
    
#data_1 = np.reshape(data,(len(distances),iterations))
data_1 = np.reshape(data,(iterations,len(distances)))
data_2 = data_1[:].T
# Now we take the probability of engulfment for 'data':
data_3 = np.zeros(len(distances))
for l in range(len(distances)):
    count = 0
    for m in range(iterations):
        if data_2[l,m] == 0:
            count += 1
        else:
            count += 0
    data_3[l] = count*100/iterations
    
    
        
# Finally we plot it girl!
x = d_cen*np.ones(len(distances))
y = distances
n = data_3

fig, ax = plt.subplots()
ax.plot(x, y,'r+')
plt.xlabel('d')
plt.ylabel('R')
plt.title('Probability of engulfment for ' + str(iterations) + ' runs')

for i, txt in enumerate(n):
    ax.annotate(int(txt), (x[i],y[i]))

plt.grid()
plt.savefig('/Users/sebas/morpheus/Results/Line_Prob_Best'+str(d_cen)+'.png')

        








































