#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:21:29 2020

@author: sebas
"""

# Surface plot of blastoid formation with respect to cell numbers.
# P1 -> 'dummy' variable
# P2 -> 'ES_pop'
# P3 -> 'TS_pop'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

#path = '/Users/sebas/morpheus/Example-CellSorting-2D_sweep_41/'
path = '/home/sebastian/morpheus/Example-CellSorting-2D_sweep_15/'

sweeps = os.listdir(path)
sweeps.remove('sweep_header.csv')
sweeps.remove('sweep_data.csv')
#sweeps.remove('.DS_Store')
sweeps.sort()

# Save number of iterations:
sweep_data = pd.read_csv(path + 'sweep_data.csv',sep='\t',header=(0))
iterations = sweep_data['P1'].unique()
number_iterations = iterations[-1] + 1 # Because dummy started on zero
# Save array with ES cel number:
ES_pop = sweep_data['P2'].unique()
# Save array with TS cell number:
TS_pop = sweep_data['P3'].unique()

# Allocate zero-tensor with dimensions of (ES_pop,TS_pop,number_iterations):
data = np.zeros(number_iterations*len(ES_pop)*len(TS_pop))

# For each iteration and cell population retrieve the value of the CT1-medium 
# boundary and store it in 'data'.
for i in range(len(sweeps)):
    sweep_path = path + sweeps[i] + '/'
    CT1_bound = pd.read_csv(sweep_path + 'logger.csv',sep='\t',header=(0))
    data[i] = CT1_bound['boundaryCT1Med'].iloc[-1]

# Save new data in a 3D format:
data_tensor = np.reshape(data,(len(iterations),len(ES_pop),len(TS_pop)))

# Now we take the probability of engulfment for 'data':starruss2014morpheus
data_final = np.zeros((len(ES_pop),len(TS_pop)))

# Fix one specific value of ES_pop and TS_pop and average over all iterations:
for i in range(len(ES_pop)):
    for j in range(len(TS_pop)):
        count = 0
        for k in range(len(iterations)):
            if data_tensor[k,i,j] == 0.0:
                count += 1
            else:
                count += 0
        data_final[i,j] = count*100/len(iterations)

data_final_flipped = np.flip(data_final,0)
# Finally and most importantly, plotting of the results:
# Visualization of resistivity
plt.figure(figsize=(12,12))
ax = sns.heatmap(data_final_flipped, linewidth=0.1)
plt.xlabel('Number of TS cells',fontsize=20)
plt.xticks(np.arange(len(TS_pop))+0.5, (TS_pop))
plt.yticks(np.arange(len(ES_pop))+0.5, (np.flip(ES_pop,0)))
#plt.setp( ax.get_yticklabels(), visible=False)
plt.ylabel('Number of ES cells',fontsize=20)
plt.title('Blastoid engulfment probability with d=0, R=100',fontsize=30)

#cbar = plt.colorbar(ax)
#cbar.ax.set_yticklabels(['0','1','2','>3'])
#cbar.set_label('Radius value', rotation=270)

plt.show()



















