# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:23:33 2019

@author: caje1277
"""

import imageRGB
import os
import matplotlib.pyplot as plt

baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-02-27 Profils électrons validation\\Doses\\'
film_dose = baseDir + 'Validation_electrons_App10x10_croix_6MeV_doseOpt_crop.tif'


img = imageRGB.load(film_dose)

plt.figure(0)
ax = plt.gca()    
img.plot(ax=ax)     
ax.plot((0, img.shape[1]), (img.center.y, img.center.y),'k--')
ax.plot((img.center.x, img.center.x), (0, img.shape[0]),'k--')
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0],0)

#%%
x0=300
y0 = 145
img.move_pixel_to_center(x0, y0)

plt.figure(1)
ax = plt.gca()    
img.plot(ax=ax)     
ax.plot((0, img.shape[1]), (img.center.y, img.center.y),'k--')
ax.plot((img.center.x, img.center.x), (0, img.shape[0]),'k--')
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0],0)

#%%
x0= 140
y0 = 325
img.move_pixel_to_center(x0, y0)

plt.figure(2)
ax = plt.gca()    
img.plot(ax=ax)     
ax.plot((0, img.shape[1]), (img.center.y, img.center.y),'k--')
ax.plot((img.center.x, img.center.x), (0, img.shape[0]),'k--')
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0],0)