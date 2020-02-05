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



# Path in ZIH computer:

# Only input should be to tell path of folder where the sweep was performed

# Path in ZIH computer:
#path = '/Users/sebas/morpheus/Example-CellSorting-2D_sweep_33/'
path = '/home/sebastian/morpheus/Example-CellSorting-2D_sweep_11/'
# Note that it works with sweep_33 in my computer and sweep_8 in ZIH computer


sweeps = os.listdir(path)
sweeps.remove('sweep_header.csv')
sweeps.remove('sweep_data.csv')
#sweeps.remove('.DS_Store')


sweeps.sort() # Get all folders sorted

# Save number of iterations:
sweep_data = pd.read_csv(path + 'sweep_data.csv',sep='\t',header=(0))
iterations = sweep_data['P1'].unique()

# Save array with scanned radii:
radii = sweep_data['P2'].unique()
# Save parameter alfa:
alfa = sweep_data['P3'].unique()
# Create zero array to store simulated data in 1D:
data = np.zeros(len(radii)*len(iterations)*len(alfa))


for i in range(len(sweeps)):
    sweep_path = path + sweeps[i] + '/'
    CT1_bound = pd.read_csv(sweep_path + 'logger.csv',sep='\t',header=(0))
    data[i] = CT1_bound['boundaryCT1Med'].iloc[-1]
    
# Save new data in a 3D format 
data_1 = np.reshape(data,(len(iterations),len(radii),len(alfa)))
#data_2 = data_1[:].T
# Now we take the probability of engulfment for 'data':
data_3 = np.zeros((len(radii),len(alfa)))

# Fix one specific value of radius,alfa and iterate over all iterations:
for i in range(len(radii)):
    for j in range(len(alfa)):
        count = 0
        for k in range(len(iterations)):
            if data_1[k,i,j] == 0:
                count += 1
            else:
                count += 0
        data_3[i,j] = count*100/len(iterations)


    
        
# Finally we plot it girl!
alfa_1 = np.zeros(len(radii)*len(alfa))
for i in range(len(radii)*len(alfa)):
    alfa_1[i] = alfa[i%len(alfa)]

x = np.array([ item for item in radii for r in range(len(alfa)) ])
y = 80 - 0.4*x + alfa_1
n = data_3.flatten() #Probabilities

fig, ax = plt.subplots()
ax.plot(x, y,'r+')
plt.xlabel('d (TS cells initial displacement from center)')
plt.ylabel('R (initial radius for TS cells distribution)')
plt.title('Engulfment probability for $r=\sqrt{ES_{pop} *50/\pi}$ (' + str(len(iterations)) + ' iterations)')

for i, txt in enumerate(n):
    ax.annotate(int(txt), (x[i],y[i]))

plt.grid()
#plt.savefig('/Users/sebas/morpheus/Results/Prob_Den for '+str(len(iterations))+'.png')
plt.savefig('/home/sebastian/Desktop/Lab_Rotation/Results/Prob_Den_dVariable_Rvariable_rSqrt_t16K.png')



#X, Y = np.meshgrid(x,y)
#
#plt.contour(X,Y,n)
#
#
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#
#fig = np.zeros((100,100))
#fig[70,0] = 84
#
#
#ax = plt.subplot(111)
#im = ax.imshow(fig)
#
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.5)
#
#plt.colorbar(im, cax=cax)




































