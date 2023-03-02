# -*- coding: utf-8 -*-
"""
Ce script est sert à démontrer exemple d'utilisation du module calibration de omg_dosimetry.
Vous pouvez en faire une copie et l'adapter selon vos besoins.
    
Écrit par Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Importer les librairies
from omg_dosimetry import calibration
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

path = os.path.join(os.path.dirname(__file__), "files", "calibration") ## Dossier racine
path_scan = os.path.join(path, "scan")                                 ## Dossier contenant les images numérisées
outname = 'Demo_calib'                                                 ## Nom du fichier de calibration à produire

#%% Définir les paramètres de calibration
#### Dose
doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]      ## Doses nominales [cGy] irradiées sur les films
output = 1.0                                          ## Si nécessaire, correction pour l'output quotidien de la machine 

### Correction latérale
lateral_correction = True                             ## True pour effectuer une calibration avec correction latérale du scanner (nécessite des longues bandes de films)
                                                         # ou False pour calibration sans correction latérale
beam_profile = os.path.join(path, "BeamProfile.txt")  ## None pour ne pas corriger pour la forme du profile de dose, 
                                                         # ou chemin d'accès vers un fichier texte contenant la forme de profile

### Détection des films
film_detect = True      ## True pour tenter une détection automatique des films, ou False pour faire une sélection manuelle
crop_top_bottom = 650   ## Si film_detect = True : Nombre de pixel à cropper dans le haut et le bas de l'image.
                           # Peut être nécessaire pour détection automatique si la vitre sur le scanner empêche la détection
roi_size = 'auto'       ## Si film_detect = True : 'auto' pour définir la taille des ROI selon les films,
                           # ou [largeur, hauteur] (mm) pour définir une taille fixe.
roi_crop = 3            ## Si film_detect = True et roi_size = 'auto' : Taille de la marge [mm] à appliquer de chaque côté
                           # des films pour définir les ROI.

### Filtre image
filt = 3                ## Taille du kernel de filtre médian à appliquer sur les images pour la réduction du bruit

#%% Produire la LUT
LUT = calibration.LUT(path=path_scan, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info, crop_top_bottom = crop_top_bottom)

#%% Afficher les résultats et sauvegarde de la LUT           
# LUT.plot_roi()  # Pour afficher les films et les ROIs utilisées pour la calibration
# LUT.plot_fit()  # Pour afficher un graphique de la courbe de calibration et la fonction algébrique fittée
LUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)            # Publication du rapport PDF
calibration.save_lut(LUT, filename=os.path.join(path, outname + '.pkl'), use_compression=True)  # Sauvegarde du fichier LUT. use_compression permet une réduction d'un facteur ~10 de la taille de fichier, mais ralentit l'opération.