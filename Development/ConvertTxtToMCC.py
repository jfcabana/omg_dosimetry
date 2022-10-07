# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 07:41:49 2019

@author: caje1277
"""
import os
import csv

#path_in = 'P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Standards\\Plane Doses\\Courbes\\'
path_in = "P:\\Projets\\CRIC\\Physique_Medicale\\Monaco\\Validation 12 MeV 20x20\\Courbes\\"
path_out = path_in + 'mcc\\'

modality = 'EL'
ssd = '1000.00'
gantry = '0.0'
#energy = '15.00'

files = os.listdir(path_in)
jj=0
for file in files:    
    fileBase, fileext = os.path.splitext(file) 
    jj=jj + 1

    print("File {} of {}: {}".format(jj, len(files), fileBase))
    
    if fileext != '.txt':
        continue  
    fileIn = path_in + fileBase + '.txt'
    fileOut = path_out + fileBase + '.mcc'
    
    if '6x6' in fileBase:
        fieldsize = '60.00'
        app = 'App6x6'    
    if '10x10' in fileBase:
        fieldsize = '100.00'
        app = 'App10x10'        
    if '14x14' in fileBase:
        fieldsize = '140.00'
        app = 'App14x14'  
    if '20x20' in fileBase:
        fieldsize = '200.00'
        app = 'App20x20'   
    if '25x25' in fileBase:
        fieldsize = '250.00'
        app = 'App25x25'   
        
    if 'PDD' in fileBase:
        scantype = 'PDD'       
    if 'PX' in fileBase:
        scantype = 'CROSSPLANE_PROFILE'      
    if 'PY' in fileBase:
        scantype = 'INPLANE_PROFILE'
    
    if '4MeV' in fileBase:
        energy = '4.00'
        depth = '10.00'
    if '6MeV' in fileBase:
        energy = '6.00'
        depth = '10.00'
    if '9MeV' in fileBase:
        energy = '9.00'
        depth = '20.00'
    if '12MeV' in fileBase:
        energy = '12.00'
        depth = '20.00'
    if '15MeV' in fileBase:
        energy = '15.00'
        depth = '20.00'
        
#    if 'G0' in fileBase:
#        gantry = '0.0'
#    if 'G330' in fileBase:
#        gantry = '330.0'       
#    if 'G340' in fileBase:
#        gantry = '340.0'     
#    if 'G350' in fileBase:
#        gantry = '350.0'
        
    # Read data
    data = []
    with open(fileIn, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data.append([row[0].replace(',','.'), row[2].replace(',','.')])

    # Write to mcc
    with open(fileOut, 'w',  newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['BEGIN_SCAN_DATA'])
        csvwriter.writerow(['','BEGIN_SCAN  1'])
        
        csvwriter.writerow(['','','COMMENT=' + fileBase])
        csvwriter.writerow(['','','MEAS_DATE=24-Jul-2019 08:00:00'])
        csvwriter.writerow(['','','MODALITY=' + modality])
        csvwriter.writerow(['','','LINAC=Monaco'])
        csvwriter.writerow(['','','ISOCENTER=-1000.00'])
        csvwriter.writerow(['','','INPLANE_AXIS_DIR=TARGET_GUN'])
        csvwriter.writerow(['','','CROSSPLANE_AXIS_DIR=LEFT_RIGHT'])
        csvwriter.writerow(['','','DEPTH_AXIS_DIR=UP_DOWN'])
        csvwriter.writerow(['','','SCAN_DEPTH=' + depth])
        csvwriter.writerow(['','','ENERGY=' + energy])
        csvwriter.writerow(['','','SSD=' + ssd])
        csvwriter.writerow(['','','SCD=0.00'])
        csvwriter.writerow(['','','WEDGE_ANGLE=0.00'])
        csvwriter.writerow(['','','WEDGE=' + app])
        csvwriter.writerow(['','','FIELD_INPLANE=' + fieldsize])
        csvwriter.writerow(['','','FIELD_CROSSPLANE=' + fieldsize])
        csvwriter.writerow(['','','GANTRY=' + gantry])
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
        csvwriter.writerow(['','','SCAN_CURVETYPE=' + scantype])
        csvwriter.writerow(['','','SCAN_OFFAXIS_INPLANE=0.00'])
        csvwriter.writerow(['','','SCAN_OFFAXIS_CROSSPLANE=0.00'])
        csvwriter.writerow(['','','SCAN_ANGLE=0.00'])
        csvwriter.writerow(['','','INCLINATION_ANGLE=0.00'])
        csvwriter.writerow(['','','DETECTOR=DOSIMETRY_DIODE'])
        csvwriter.writerow(['','','DETECTOR_NAME=Monaco'])
        csvwriter.writerow(['','','BEGIN_DATA'])
    
        for i in range(0, len(data)):
             csvwriter.writerow(['','','', float(data[i][0])*10, data[i][1]])
    
        csvwriter.writerow(['','','END_DATA'])
        csvwriter.writerow(['','END_SCAN  1'])
        csvwriter.writerow(['END_SCAN_DATA'])