# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import tiff2dose
import os

#%% Set info
info = dict(author = 'JMe',
            unit = 'Salle 1',
            film_lot = 'C11',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2022-03-03',
            date_scanned = '2022-03-04',
            wait_time = '18 heures',
            notes = 'Scan en transmission @300ppp, avec la vitre'
           )
# Set paths
# ** Atention, si un nom de répertoire commence avec un chiffre, mettre un double \ devant **
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Cones\\3_Modelisation\\Données brutes et Analyse\\Films_Analyse\\MLC_10x10-2'
path_in = path + '\\2'

#lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C11-XD_calib_18h_trans_300ppp_0-30Gy.pkl'
lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C10_calib_24h_trans_vitre_0-10Gy.pkl'
fit_type = 'rational'

ext = '_' + fit_type

path_out = os.path.join(path, 'Dose')

# Set options
clip = 3000
rot90 = 0

# Perform tiff2dose conversion
if not os.path.exists(path_out):
    os.makedirs(path_out)

files = os.listdir(path_in)

for file in files:
    
    img_file = os.path.join(path_in, file)
    filebase, fileext = os.path.splitext(file)
    
    if img_file[-7:] != '001.tif':
        continue
    if os.path.isdir(img_file):
        continue
    
    gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type=fit_type, info=info, clip = clip, rot90=rot90)
#    gaf1.dose_opt.crop_edges(threshold=60)
#    gaf1.dose_m.save(os.path.join(path_out, filebase[:-4] + ext + '_m.tif'))
#    gaf1.dose_r.save(os.path.join(path_out, filebase[:-4] + ext + '_r.tif'))
#    gaf1.dose_g.save(os.path.join(path_out, filebase[:-4] + ext + '_g.tif'))
#    gaf1.dose_b.save(os.path.join(path_out, filebase[:-4] + ext + '_b.tif'))
    gaf1.dose_rg.save(os.path.join(path_out, filebase[:-4] + ext + '_rg.tif'))
#    gaf1.dose_ave.save(os.path.join(path_out, filebase[:-4] + ext + '_ave.tif'))
    gaf1.dose_opt.save(os.path.join(path_out, filebase[:-4] + ext + '_opt.tif'))
    
    report_doseFilm = os.path.join(path_out, filebase[:-4] + ext + '.pdf')
    gaf1.publish_pdf(filename=report_doseFilm, open_file=False)