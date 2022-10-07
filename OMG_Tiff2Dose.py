# -*- coding: utf-8 -*-
"""
Utiliser ce fichier pour convertir vos scans en dose absolue. Modifier les informations pour correspondre à vos besoins.
"""


import tiff2dose
import os

#%% Set info
info = dict(author = 'JMe',
            unit = 'Salle 1',
            film_lot = 'C10',
            scanner_id = 'Epson 12000XL',
            date_exposed = '20211220',
            date_scanned = '20211221',
            wait_time = '24 hr',
            notes = 'Scan en transmission avec la vitre.'
           )

#%% Set paths ** Atention, si un nom de répertoire commence avec un chiffre, mettre un double \ devant **

basePath = 'P:/Projets/CRIC/Physique_Medicale/SRS/Commissioning/Cones/3_Mesures-Modelisation/Films/C09mm/1'
path_in = os.path.join(basePath)
path_out = os.path.join(basePath, 'Analyse')

#lut_file = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\\\2020-12-14 Calibration C10 SRS\\C10_calib_24h_trans_vitre_0-10Gy.pkl'
lut_file = 'P:\Projets\CRIC\Physique_Medicale\Films\Calibrations\C10_calib_24h_trans_vitre_0-10Gy.pkl'

#%% Set options
clip = 700     # Entrer une valeur en cGy maximum où couper la dose. Si 'None', n'applique pas de threshold.
rot90 = 0       # Au besoin, mettre à 1 pour appliquer une rotation de 90 degrés sur l'image (si l'image de scan est verticale)
fit_type = 'rational'   # 'spline' ou 'rational'. Laisser à 'rational' par défaut.

#%% Perform tiff2dose conversion
if not os.path.exists(path_out):
    os.makedirs(path_out)

files = os.listdir(path_in)

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

    gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type=fit_type, info=info, clip = clip, rot90=rot90)
#    gaf1 = tiff2dose.Gaf(path=path_in, lut_file=lut_file, fit_type=fit_type, info=info, clip = clip, rot90=rot90)
    gaf1.dose_opt.save(os.path.join(path_out, filebase + '_' + fit_type + '.tif'))
#    gaf1.publish_pdf(filename=os.path.join(path_out, filebase + '_report.pdf'), open_file=True)