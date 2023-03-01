# -*- coding: utf-8 -*-
"""
Ce script est sert à démontrer l'utilisation du module tiff2dose de omg_dosimetry.
Vous pouvez en faire une copie et l'adapter selon vos besoins.
    
Écrit par Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Importer les librairies
from omg_dosimetry import tiff2dose
import os

#%% Définir les informations générales
info = dict(author = 'Demo Physicien',
            unit = 'Demo Linac',
            film_lot = 'XD_1',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-01-24 16h',
            date_scanned = '2023-01-25 16h',
            wait_time = '24 heures',
            notes = 'Scan en transmission @300ppp'
           )

path = os.path.join(os.path.dirname(__file__), "files", "tiff2dose")   # Dossier racine
path_scan = os.path.join(path, "scan")                                 # Dossier contenant les images numérisées
outname = 'Demo_calib'                                                 # Nom du fichier de calibration à produire



#%% Set info
info = dict(author = 'JFC',
            unit = 'Salle 2',
            film_lot = 'C14 XD',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-01-24 16h',
            date_scanned = '2023-01-25 10h',
            wait_time = '18 heures',
            notes = 'Scan en transmission @300ppp')
# Set paths
# ** Atention, si un nom de répertoire commence avec un chiffre, mettre un double \ devant **
path = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibrations\\2023-01-23 C14 XD\\18h\\Valid"
path = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\QA Patients\\0phy_SRS_multi\\Test"
path_in = path + '\\Scan'
lut_file = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibrations\\C14_calib_18h_trans_300ppp_0-9Gy.pkl"
fit_type = 'rational'
ext = '_0-9Gy'
path_out = os.path.join(path, 'Dose' + ext)

# Set options
clip = 3000
rot90 = 0

# Perform tiff2dose conversion
if not os.path.exists(path_out): os.makedirs(path_out)
    
files = os.listdir(path_in)
for file in files:
    img_file = os.path.join(path_in, file)
    filebase, fileext = os.path.splitext(file)
    if img_file[-7:] != '001.tif': continue
    if os.path.isdir(img_file): continue
    
    gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type=fit_type, info=info, clip = clip, rot90=rot90)
    gaf1.dose_opt.save(os.path.join(path_out, filebase[:-4] + ext + '_opt.tif')) 
    report_doseFilm = os.path.join(path_out, filebase[:-4] + ext + '.pdf')
    gaf1.publish_pdf(filename=report_doseFilm, open_file=True)