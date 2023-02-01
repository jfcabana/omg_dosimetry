# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:41:59 2023

@author: caje1277
"""

from pylinac.core.profile import SingleProfile, MultiProfile
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
plt.ion()
import numpy as np
#from System.Threading import Thread, ThreadStart
#from System.Windows import *

path = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\QA Patients\\0phy_SRS_multi\\Old\\A1A_Multi_2cm"

file_pkl = os.path.join(path, "Analyse_CC.pkl")
with open(file_pkl, 'rb') as f:
    film_CC = pickle.load(f)

file_pkl = os.path.join(path, "Analyse_MC.pkl")
with open(file_pkl, 'rb') as f:
    film_MC = pickle.load(f)

film_CC.show_results()
film_MC.show_results()
##%%
#def SetText(text):
#    def thread_proc():
#        System.Windows.Forms.Clipboard.SetText(text)
#
#    t = Thread(ThreadStart(thread_proc))
#    t.ApartmentState = System.Threading.ApartmentState.STA
#    t.Start()
#    
#%% Analyse différence RS vs film
img = film.ref_dose
if '3x3' in path: roi_size = 100
elif '1x1' in path: roi_size = 20
# On trouve les positions x 
thresh=0.5
row = np.mean(img.array, axis=-1)
bined = np.where(row > max(row) * thresh, 1, 0)    # binarize the profile for improved detectability of peaks
x_profile = MultiProfile(bined)
x = x_profile.find_fwxm_peaks(x=50, threshold=0.3, min_distance=0.02)
xpos = [int(i) for i in x]
#x_profile.plot()

# On trouve les positions y 
col = np.mean(img.array, axis=0)
bined = np.where(col > max(col) * thresh, 1, 0)    # binarize the profile for improved detectability of peaks
y_profile = MultiProfile(bined)
y = y_profile.find_fwxm_peaks(x=50, threshold=0.3, min_distance=0.02)
ypos = [int(i) for i in x]
#y_profile.plot()

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10, 4))
img.plot(ax1, title="RS")
film.film_dose.plot(ax2, title="Film")
doses_ref = []
doses_film = []
doses_diff = []
#report = ''
for x in xpos:
    for y in ypos:
        x1 = x-roi_size
        x2 = x+roi_size
        y1 = y-roi_size
        y2 = y+roi_size
        dose_ref = round(np.mean(img.array[x1:x2, y1:y2]),1)
        dose_film = round(np.mean(film.film_dose.array[x1:x2, y1:y2]),1)
        dose_diff = round((dose_ref - dose_film) / dose_film * 100,1)
        doses_ref.append(dose_ref)
        doses_film.append(dose_film)
        doses_diff.append(dose_diff)
        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), Fill=False )
        rect2 = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), Fill=False )
        ax1.add_patch(rect)
        ax2.add_patch(rect2)
#        report += dose_diff + '\t'
#    report += '\n'
print("======== Dose film [cGy] ========")   
print(np.array(doses_film).reshape(len(xpos),len(ypos)))
print("======== Dose RS [cGy] ========")   
print(np.array(doses_ref).reshape(len(xpos),len(ypos)))
print("======== Diff RS - film [%] ========")   
print(np.array(doses_diff).reshape(len(xpos),len(ypos)))
#SetText(report)
