# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:41:59 2023

@author: caje1277
"""

import os
import pickle
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import csv

path = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\QA Patients\\0phy_SRS_multi\\"
subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]

for folder in subfolders:
    if 'Norm' in folder: continue
    # if 'A1A_1x1_2_10cm' not in folder: continue
    file_pkl = os.path.join(folder, "Analyse.pkl")
    if not os.path.isfile(file_pkl): continue
    with open(file_pkl, 'rb') as f:
        film = pickle.load(f)
    fileBase = folder.split("\\")[-1]
    
    x0 = int(film.film_dose.center.x)
    y0 = int(film.film_dose.center.y)
    width = 5
    crop = 50
            
    img = film.film_dose.array
    crossline_prof = np.median(img[y0-width:y0+width, crop:-1-crop], axis=0)
    crossline_pos = (np.asarray(range(len(crossline_prof))) - int(len(crossline_prof)/2)) / film.film_dose.dpmm
    
    inline_prof = np.median(img[crop:-1-crop, x0-width:x0+width], axis=1)
    inline_pos = (np.asarray(range(len(inline_prof))) - int(len(inline_prof)/2)) / film.film_dose.dpmm
    
    # Save profiles
    #    plt.plot(inline_pos, inline_prof,crossline_pos, crossline_prof)
    fileOut = os.path.join(path,fileBase + ".png")
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10, 4))
    film.plot_profile(ax=ax1, profile='x', title='Horizontal profile (y={})'.format(y0), position=y0)
    film.plot_profile(ax=ax2, profile='y', title='Vertical profile (x={})'.format(x0), position=x0)
    fig.savefig(fileOut)
    # plt.close(fig)

    
    # Write to mcc
    fileOut = os.path.join(path,fileBase + ".mcc")
    if "1x1" in fileBase: FS = "10.0"
    if "3x3" in fileBase: FS = "30.0"
    if '2cm' in fileBase: prof = "20.0"
    if '6cm' in fileBase: prof = "60.0"
    if '10cm' in fileBase: prof = "100.0"
    
    with open(fileOut, 'w',  newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['BEGIN_SCAN_DATA'])
        csvwriter.writerow(['','BEGIN_SCAN  1'])
        
        csvwriter.writerow(['','','COMMENT=EBT3 '+ fileBase])
        csvwriter.writerow(['','','MEAS_DATE=20-Dec-2021 00:00:00'])
        csvwriter.writerow(['','','MODALITY=X'])
        csvwriter.writerow(['','','ISOCENTER=1000.00'])
        csvwriter.writerow(['','','INPLANE_AXIS=Inplane'])
        csvwriter.writerow(['','','CROSSPLANE_AXIS=Crossplane'])
        csvwriter.writerow(['','','DEPTH_AXIS=100'])
        csvwriter.writerow(['','','INPLANE_AXIS_DIR=TARGET_GUN'])
        csvwriter.writerow(['','','CROSSPLANE_AXIS_DIR=LEFT_RIGHT'])
        csvwriter.writerow(['','','DEPTH_AXIS_DIR=UP_DOWN'])
        csvwriter.writerow(['','','ENERGY=10.00'])
        csvwriter.writerow(['','','SSD=940.00'])
        csvwriter.writerow(['','','SCD=0.00'])
        csvwriter.writerow(['','','WEDGE_ANGLE=0.00'])
        csvwriter.writerow(['','','FIELD_INPLANE=' + FS])
        csvwriter.writerow(['','','FIELD_CROSSPLANE=' + FS])
        csvwriter.writerow(['','','FIELD_TYPE=RECTANGULAR'])
        csvwriter.writerow(['','','GANTRY=0.00'])
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
        csvwriter.writerow(['','','SCAN_OFFAXIS_INPLANE=0.00'])
        csvwriter.writerow(['','','SCAN_OFFAXIS_CROSSPLANE=0.00'])
        csvwriter.writerow(['','','SCAN_ANGLE=0.00'])
        csvwriter.writerow(['','','INCLINATION_ANGLE=0.00'])
        csvwriter.writerow(['','','DETECTOR_NAME=Film-C13'])
        csvwriter.writerow(['','','FILTER=FFF'])
            
        csvwriter.writerow(['','','SCAN_CURVETYPE=INPLANE_PROFILE'])
        csvwriter.writerow(['','','SCAN_DEPTH=' + prof])
        csvwriter.writerow(['','','BEGIN_DATA'])
    
        for i in range(0, len(inline_prof)):
             if inline_prof[i] == 0:
                 continue
             csvwriter.writerow(['','','', inline_pos[i], inline_prof[i]])
    
        csvwriter.writerow(['','','END_DATA'])
        csvwriter.writerow(['','END_SCAN  1'])
        
        csvwriter.writerow(['','BEGIN_SCAN  2'])
        
        csvwriter.writerow(['','','COMMENT=EBT3 ' + fileBase])
        csvwriter.writerow(['','','MEAS_DATE=20-Dec-2021 00:00:00'])
        csvwriter.writerow(['','','MODALITY=X'])
        csvwriter.writerow(['','','ISOCENTER=1000.00'])
        csvwriter.writerow(['','','INPLANE_AXIS=Inplane'])
        csvwriter.writerow(['','','CROSSPLANE_AXIS=Crossplane'])
        csvwriter.writerow(['','','DEPTH_AXIS=Profondeur'])
        csvwriter.writerow(['','','INPLANE_AXIS_DIR=TARGET_GUN'])
        csvwriter.writerow(['','','CROSSPLANE_AXIS_DIR=LEFT_RIGHT'])
        csvwriter.writerow(['','','DEPTH_AXIS_DIR=UP_DOWN'])
        csvwriter.writerow(['','','ENERGY=10.00'])
        csvwriter.writerow(['','','SSD=940.00'])
        csvwriter.writerow(['','','SCD=0.00'])
        csvwriter.writerow(['','','WEDGE_ANGLE=0.00'])
        csvwriter.writerow(['','','FIELD_INPLANE=' + FS])
        csvwriter.writerow(['','','FIELD_CROSSPLANE=' + FS])
        csvwriter.writerow(['','','FIELD_TYPE=RECTANGULAR'])
        csvwriter.writerow(['','','GANTRY=0.00'])
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
        csvwriter.writerow(['','','SCAN_OFFAXIS_INPLANE=0.00'])
        csvwriter.writerow(['','','SCAN_OFFAXIS_CROSSPLANE=0.00'])
        csvwriter.writerow(['','','SCAN_ANGLE=0.00'])
        csvwriter.writerow(['','','INCLINATION_ANGLE=0.00'])
        csvwriter.writerow(['','','DETECTOR_NAME=Film-C13'])
        csvwriter.writerow(['','','FILTER=FFF'])
        
        csvwriter.writerow(['','','SCAN_CURVETYPE=CROSSPLANE_PROFILE'])
        csvwriter.writerow(['','','SCAN_DEPTH=' + prof])
        csvwriter.writerow(['','','BEGIN_DATA'])
        
        for i in range(0, len(crossline_prof)):
             if crossline_prof[i] == 0:
                 continue
             csvwriter.writerow(['','','', crossline_pos[i], crossline_prof[i]])
        
        csvwriter.writerow(['','','END_DATA'])
        csvwriter.writerow(['','END_SCAN  2'])
        csvwriter.writerow(['END_SCAN_DATA'])
        