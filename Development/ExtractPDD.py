# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

#%% Set paths

#baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2020-12-14 Calibration C10 SRS\\Valid 4h\\'
baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Mesures\\Comparaison détecteurs\\Films\\Rescan PDD\\Scan 300ppp\\'

path_in = baseDir + 'Dose_spline\\'
path_out = baseDir + 'MCC\\'

if not os.path.exists(path_out):
    os.makedirs(path_out)

#files = os.listdir(path_in)

fileBase = 'C10-SRS_PDD_refl_300ppp_5-2_spline_g'
fileIn = path_in + fileBase + '.tif'
fileOut = path_out + fileBase + '.mcc'

film = analysis.DoseAnalysis(film_dose=fileIn, flipLR=True, ref_dose=None, rot90=0, ref_dose_factor=None)

#Sélectionner les marqueurs sur l'image en double cliquant.
film.register(threshold=10)

#%%

width = 5  # number of pixels on each side of center to get average profile
crop = 0

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

#img = film.film_dose.array / film.film_dose.array[y0][x0] 
inline_prof = np.median(film.film_dose.array[y0-width:y0+width,:], axis=0)
inline_pos = (np.asarray(range(len(inline_prof))) - int(len(inline_prof)/2)) / film.film_dose.dpmm

crossline_prof = np.median(film.film_dose.array[:, x0-width:x0+width], axis=1)
crossline_pos = (np.asarray(range(len(crossline_prof))) - int(len(crossline_prof)/2)) / film.film_dose.dpmm

profile = inline_prof
#position = (np.asarray(range(len(profile)))) / film.film_dose.dpmm + 100
position = inline_pos + 100

plt.figure()
plt.plot(position, profile)

 # Write to mcc
with open(fileOut, 'w',  newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['BEGIN_SCAN_DATA'])
    csvwriter.writerow(['','BEGIN_SCAN  1'])
    
    csvwriter.writerow(['','','COMMENT=' + fileBase])
    csvwriter.writerow(['','','MEAS_DATE=12-Nov-2019 08:00:00'])
    csvwriter.writerow(['','','MODALITY=X'])
    csvwriter.writerow(['','','ISOCENTER=-1000.00'])
    csvwriter.writerow(['','','INPLANE_AXIS_DIR=TARGET_GUN'])
    csvwriter.writerow(['','','CROSSPLANE_AXIS_DIR=LEFT_RIGHT'])
    csvwriter.writerow(['','','DEPTH_AXIS_DIR=UP_DOWN'])
    csvwriter.writerow(['','','ENERGY=6.00'])
    csvwriter.writerow(['','','SSD=900.00'])
    csvwriter.writerow(['','','SCD=0.00'])
    csvwriter.writerow(['','','WEDGE_ANGLE=0.00'])
    csvwriter.writerow(['','','FIELD_INPLANE=100.00'])
    csvwriter.writerow(['','','FIELD_CROSSPLANE=100.00'])
    csvwriter.writerow(['','','GANTRY=90.00'])
    csvwriter.writerow(['','','GANTRY_UPRIGHT_POSITION=0'])
    csvwriter.writerow(['','','GANTRY_ROTATION=CW'])
    csvwriter.writerow(['','','COLL_ANGLE=0.00'])
    csvwriter.writerow(['','','COLL_OFFSET_INPLANE=0.00'])
    csvwriter.writerow(['','','COLL_OFFSET_CROSSPLANE=0.00'])
    csvwriter.writerow(['','','SCAN_DEVICE_SETUP=BARA_LEFT_RIGHT'])
    csvwriter.writerow(['','','DETECTOR_IS_CALIBRATED=0'])
    csvwriter.writerow(['','','DETECTOR_REFERENCE_IS_CALIBRATED=0'])
    csvwriter.writerow(['','','REF_FIELD_DEPTH=0.00'])
    csvwriter.writerow(['','','REF_FIELD_DEFINED=WATER_SURFACE'])
    csvwriter.writerow(['','','SCAN_CURVETYPE=PDD'])
    csvwriter.writerow(['','','SCAN_OFFAXIS_INPLANE=0.00'])
    csvwriter.writerow(['','','SCAN_OFFAXIS_CROSSPLANE=0.00'])
    csvwriter.writerow(['','','SCAN_ANGLE=0.00'])
    csvwriter.writerow(['','','INCLINATION_ANGLE=0.00'])
    csvwriter.writerow(['','','DETECTOR_NAME=Film-C10'])
    csvwriter.writerow(['','','BEGIN_DATA'])

    for i in range(0, len(profile)):
         if profile[i] == 0:
             continue
         csvwriter.writerow(['','','', position[i], profile[i]])

    csvwriter.writerow(['','','END_DATA'])
    csvwriter.writerow(['','END_SCAN  1'])
    csvwriter.writerow(['END_SCAN_DATA'])
