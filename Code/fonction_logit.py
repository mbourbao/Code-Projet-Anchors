# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:13:21 2021

@author: 33651
"""

import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.patches as mpatches
f= lambda x : 1/(1+np.exp(-x))

x= np.arange(-100,100,1)

y = f(x)
plt.plot(x,y,color="red")
red_patch = mpatches.Patch(color='red', label='Fonction logistique')
plt.legend(handles=[red_patch])
plt.title('Tracer de la fonction logistique') 
