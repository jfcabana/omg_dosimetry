# -*- coding: utf-8 -*-
"""
"""

from omg_dosimetry import analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

#% Set paths
baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Mesures\\2023-03-09 Eval scanner\\Moyennage\Dose\\'
path_in = baseDir
path_out = baseDir + 'MCC\\'
if not os.path.exists(path_out): os.makedirs(path_out)

files = os.listdir(path_in)
for file in files:
    fileBase, fileext = os.path.splitext(file)
    if fileext != '.tif': continue

    fileIn = os.path.join(path_in, file)
    fileOut = os.path.join(path_out, fileBase + '.mcc')
    film = analysis.DoseAnalysis(film_dose=fileIn, flipLR=True, ref_dose=fileIn, rot90=1)
    
    FS = '250'
    width = 3  # number of pixels on each side of center to get average profile
    crop = 1
    Offset_X = 0  #pixels offset profils inplanes Y
    Offset_Y = 0  #pixels offset profils crossplane X
    
    x0 = int(film.film_dose.array.shape[1]/2) + Offset_X
    y0 = int(film.film_dose.array.shape[0]/2) + Offset_Y
    
    img = film.film_dose.array / film.film_dose.array[y0][x0]
    
    crossline_prof = np.median(img[y0-width:y0+width, crop:-1*crop], axis=0)
    crossline_pos = (np.asarray(range(len(crossline_prof))) - int(len(crossline_prof)/2)) / film.film_dose.dpmm
    
    inline_prof = np.median(img[crop:-1*crop, x0-width:x0+width], axis=1)
    inline_pos = (np.asarray(range(len(inline_prof))) - int(len(inline_prof)/2)) / film.film_dose.dpmm
    
    # Write to mcc
    fieldnames = ['Position', 'Value']
    
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
        csvwriter.writerow(['','','ENERGY=6.00'])
        csvwriter.writerow(['','','SSD=900.00'])
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
        csvwriter.writerow(['','','DETECTOR_NAME=Film-C10'])
        csvwriter.writerow(['','','FILTER=FFF'])
            
        csvwriter.writerow(['','','SCAN_CURVETYPE=INPLANE_PROFILE'])
        csvwriter.writerow(['','','SCAN_DEPTH=100.00'])
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
        csvwriter.writerow(['','','ENERGY=6.00'])
        csvwriter.writerow(['','','SSD=900.00'])
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
        csvwriter.writerow(['','','DETECTOR_NAME=Film-C10'])
        csvwriter.writerow(['','','FILTER=FFF'])
        
        csvwriter.writerow(['','','SCAN_CURVETYPE=CROSSPLANE_PROFILE'])
        csvwriter.writerow(['','','SCAN_DEPTH=100.00'])
        csvwriter.writerow(['','','BEGIN_DATA'])
        
        for i in range(0, len(crossline_prof)):
             if crossline_prof[i] == 0:
                 continue
             csvwriter.writerow(['','','', crossline_pos[i], crossline_prof[i]])
        
        csvwriter.writerow(['','','END_DATA'])
        csvwriter.writerow(['','END_SCAN  2'])
        csvwriter.writerow(['END_SCAN_DATA'])
        
