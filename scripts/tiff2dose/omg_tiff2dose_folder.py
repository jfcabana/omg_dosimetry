# -*- coding: utf-8 -*-
"""
Conversion de films scannés en dose.
    
Écrit par Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Importer les librairies
from omg_dosimetry import tiff2dose
import os

#%% Définir les informations générales
info = dict(author = 'JFC',
            unit = 'VHD2',
            film_lot = 'C14',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-03-09 14h',
            date_scanned = '2023-03-10 14h',
            wait_time = '24 heures',
            notes = 'Scan en transmission @300ppp'
           )

path_films = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\'
path_in = os.path.join(path_films, "Mesures","2023-03-09 Eval scanner", "Moyennage")   # Dossier racine
path_scan = os.path.join(path_in, "Scan")                                 # Dossier contenant les images numérisées, ou nom de fichier si le dossier contient plusieurs numérisations de films différents
path_out = os.path.join(path_in, "Dose")                                 # Dossier de sortie

#%% Définir les paramètres de conversion en dose
path_calib = os.path.join(path_films, 'Calibrations')
lut_file = os.path.join(path_calib, 'C14_calib_18h_trans_300ppp_0-30Gy.pkl')   # Chemin vers le fichier LUT à utiliser
fit_type = 'rational'                                                                           # Type de fonction à utiliser pour le fit de la courbe de calibration. 'rational' ou 'spline'
clip = 3000                                                                                      # Valeur maximale [cGy] à laquelle la dose doit être limitée. Utile pour éviter des valeurs de dose extrêmes par exemple sur les marques faites sur le film.

#%% Effectuer la conversion en dose

# Perform tiff2dose conversion
if not os.path.exists(path_out): os.makedirs(path_out)

files = os.listdir(path_scan)
for file in files:
    img_file = os.path.join(path_scan, file)
    filebase, fileext = os.path.splitext(file)
    if img_file[-7:] != '001.tif': continue
    outname = filebase[:-4]

    gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type=fit_type, info=info, clip=clip)

    # Sauvegarder la dose et produire le rapport PDF
    filename_tif = os.path.join(path_out, outname+'.tif')
    gaf1.dose_opt.save(filename_tif)                    # On sauvegarde la "dose_opt". D'autres options sont disponibles également.
    
    # filename_pdf = os.path.join(path_out, outname+'.pdf')
    # gaf1.publish_pdf(filename_pdf, open_file=False)      # Publication du rapport PDF