# -*- coding: utf-8 -*-
"""
Utiliser ce fichier pour convertir vos scans en dose absolue. Modifier les informations pour correspondre à vos besoins.
"""

import tiff2dose
import os

#%% Set info
info = dict(author = 'JFC',
            unit = 'Salle 1',
            film_lot = 'C6-XD',
            scanner_id = 'Epson 12000XL',
            date_exposed = '2019/05/30',
            date_scanned = '2019/05/31',
            wait_time = '24 hr',
            notes = 'Scan en transmission.'
           )

#%% Set paths

# ** Atention, si un nom de répertoire commence avec un chiffre, mettre un double \ devant **

basePath = 'P:\Projets\CRIC\Physique_Medicale\Films\QA Patients\\385575\ThLSD_1A'

path_in = os.path.join(basePath, 'Scan')
path_out = os.path.join(basePath, 'DoseFilm')

lut_path = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations'
lut_file =  os.path.join(lut_path, 'C6-XD_calib_24h_trans_LatCorr_0-24Gy_vitre.pkl')

if not os.path.exists(path_out):
    os.makedirs(path_out)

#%% Set options
clip = None
rot90 = 0
threshold=60
fit_type = 'rational'

#%% Perform tiff2dose conversion

files = os.listdir(path_in)

ext = ''


for file in files:
    
    img_file = os.path.join(path_in, file)
    filebase, fileext = os.path.splitext(file)
    
    if file == 'Thumbs.db':
        continue
    if os.path.isdir(img_file):
        continue
    if filebase[-4:-1] == '_00':
        if filebase[-1] != '1':
            continue
        filebase = filebase[0:-4]
      
    gaf1 = tiff2dose.Gaf(path=path_in, lut_file=lut_file, fit_type=fit_type, info=info, clip = clip, rot90=rot90)
    gaf1.dose_opt.crop_edges(threshold=threshold)
    gaf1.dose_opt.save(os.path.join(path_out, filebase + '_doseOpt.tif'))
    gaf1.publish_pdf(filename=os.path.join(path_out, filebase +'_report.pdf'), open_file=True)