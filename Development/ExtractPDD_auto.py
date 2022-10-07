# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import medfilt

#%% Set paths

baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-11-12 Calibration C9 XD\\Verif position scan PDD\\'

path_in = baseDir + 'DoseFilm\\'
path_out = baseDir + 'Rendements\\'

if not os.path.exists(path_out):
    os.makedirs(path_out)

files = os.listdir(path_in)

film_filt = 15

for file in files:
    fileBase, fileext = os.path.splitext(file)   
    if fileext != '.tif':
        continue

    fileIn = path_in + fileBase + '.tif'
    fileOut = path_out + fileBase + '_f15.mcc'
    
    film = analysis.DoseAnalysis(film_dose=fileIn, flipLR=True, ref_dose_factor=None)
    if film_filt >0 :
        film.film_dose.array = medfilt(film.film_dose.array,  kernel_size=(film_filt, film_filt))
    film.film_dose.crop_edges(threshold=10)

    width = 9  # number of pixels on each side of center to get average profile
    crop = 0
    
    x0 = int(film.film_dose.shape[1] / 2)
    y0 = int(film.film_dose.shape[0] / 2)
    
    inline_prof = np.median(film.film_dose.array[y0-width:y0+width,:], axis=0)
    inline_pos = (np.asarray(range(len(inline_prof)))) / film.film_dose.dpmm
    
    crossline_prof = np.median(film.film_dose.array[:, x0-width:x0+width], axis=1)
    crossline_pos = (np.asarray(range(len(crossline_prof))) - int(len(crossline_prof)/2)) / film.film_dose.dpmm
    
    profile = inline_prof
    position = (np.asarray(range(len(profile)))) / film.film_dose.dpmm
        
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
        csvwriter.writerow(['','','DETECTOR_NAME=Film-C9-XD'])
        csvwriter.writerow(['','','BEGIN_DATA'])
    
        for i in range(0, len(profile)):
#             if inline_prof[i] == 0:
#                 continue
             csvwriter.writerow(['','','', position[i], profile[i]])
    
        csvwriter.writerow(['','','END_DATA'])
        csvwriter.writerow(['','END_SCAN  1'])
        csvwriter.writerow(['END_SCAN_DATA'])
