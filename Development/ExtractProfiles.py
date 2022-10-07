# -*- coding: utf-8 -*-
"""
"""

import analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

#%% Set paths

baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-03-25 Calibration lot 04\\Valid\\'

path_in = baseDir + 'ConvertedDoses1\\'
path_out = baseDir + 'Profiles1\\'

if not os.path.exists(path_out):
    os.makedirs(path_out)

#files = os.listdir(path_in)

#%%

fileBase = 'C4_valid1_doseOpt'
fileIn = path_in + fileBase + '.tif'
fileOut = path_out + fileBase + '.mcc'

film = analysis.DoseAnalysis(film_dose=fileIn)

#Sélectionner les marqueurs sur l'image en double cliquant.
film.register(threshold=100)

#%%

width = 9  # number of pixels on each side of center to get average profile
crop = 5

x0 = int(film.film_dose.shape[1] / 2)
y0 = int(film.film_dose.shape[0] / 2)

#x0 = film.x0
#y0 = film.y0

x1 = film.markers[0][0]
y1 = film.markers[0][1]
x2 = film.markers[1][0]
y2 = film.markers[1][1]
x3 = film.markers[2][0]
y3 = film.markers[2][1]
x4 = film.markers[3][0]
y4 = film.markers[3][1]

#x0 = int((x1 + x3)/2)
#y0 = int((y2 + y4)/2)

img = film.film_dose.array / film.film_dose.array[y0][x0]
#
#start = x4-length
#stop = x2+length
#if start < 0:
#    start = 0
#if stop > img.shape[1]:
#    stop = img.shape[1]-1

inline_prof = np.median(img[y0-width:y0+width, crop:-1*crop], axis=0)
inline_pos = (np.asarray(range(len(inline_prof))) - int(len(inline_prof)/2)) / film.film_dose.dpmm


#start = y1-length
#stop = y3+length
#if start < 0:
#    start = 0
#if stop > img.shape[0]:
#    stop = img.shape[0]-1
    
crossline_prof = np.median(img[crop:-1*crop, x0-width:x0+width], axis=1)
crossline_pos = (np.asarray(range(len(crossline_prof))) - int(len(crossline_prof)/2)) / film.film_dose.dpmm

plt.figure()
plt.plot(inline_pos, inline_prof)
plt.figure()
plt.plot(crossline_pos, crossline_prof)

# Write to mcc
fieldnames = ['Position', 'Value']

with open(fileOut, 'w',  newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['BEGIN_SCAN_DATA'])
    csvwriter.writerow(['','BEGIN_SCAN  1'])
    csvwriter.writerow(['','','SCAN_CURVETYPE=INPLANE_PROFILE'])
    csvwriter.writerow(['','','BEGIN_DATA'])

    for i in range(0, len(inline_prof)):
         if inline_prof[i] == 0:
             continue
         csvwriter.writerow(['','','', inline_pos[i], inline_prof[i]])

    csvwriter.writerow(['','','END_DATA'])
    csvwriter.writerow(['','END_SCAN  1'])
    
    csvwriter.writerow(['','BEGIN_SCAN  2'])
    csvwriter.writerow(['','','SCAN_CURVETYPE=CROSSPLANE_PROFILE'])
    csvwriter.writerow(['','','BEGIN_DATA'])
    
    for i in range(0, len(crossline_prof)):
         if crossline_prof[i] == 0:
             continue
         csvwriter.writerow(['','','', crossline_pos[i], crossline_prof[i]])
    
    csvwriter.writerow(['','','END_DATA'])
    csvwriter.writerow(['','END_SCAN  2'])
    csvwriter.writerow(['END_SCAN_DATA'])
