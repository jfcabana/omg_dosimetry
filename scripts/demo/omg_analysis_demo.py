# -*- coding: utf-8 -*-
"""
Ce script est sert à démontrer un exemple d'utilisation du module analysis de omg_dosimetry.
Vous pouvez en faire une copie et l'adapter selon vos besoins.
    
Écrit par Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Importer les librairies
from omg_dosimetry import analysis
import os
    
#%% Définir les chemins d'accès aux fichiers
path_analysis = os.path.join(os.path.dirname(__file__), "files", "analysis")                   # Dossier racine pour l'analyse
ref_dose = os.path.join(path_analysis, 'DoseRS')                                               # Dossier contenant la dose planaire de référence
file_doseFilm = os.path.join(os.path.dirname(__file__), "files", "tiff2dose", "Demo_dose.tif") # Chemin vers la dose film (doit avoir exécuté omg_tiff2dose_demo en premier)
filebase = 'Demo_analysis'                                                                     # Nom du fichier de rapport à produire

#%% Définir les paramètres de l'analyse
#### Facteurs de normalisation
ref_dose_factor = 1.0   # Si nécessaire, appliquer un facteur de normalisation à la dose de référence
film_dose_factor = 1.0  # Si nécessaire, appliquer un facteur de normalisation à la dose film
prescription = 300      # Dose de prescription

#### Paramètres de recalage 
flipLR = True           # True/False: Appliquer une transformation miroir horizontale à la dose film
flipUD = False          # True/False: Appliquer une transformation miroir verticale à la dose film
rot90 = 1               # int: Nombre de rotations de 90 degrés à appliquer à la dose film
shift_x = 0.0           # Si nécessaire, valeur connue du décalage [mm] à appliquer au film en direction X
shift_y = -0.8          # Si nécessaire, valeur connue du décalage [mm] à appliquer au film en direction Y
markers_center = [0.8, 1.2, 233.3]  # Coordonnées dans le fantôme qui correspond au centre des marques faites sur le film

#### Paramètres de l'analyse Gamma
threshold = 0.20         # Seuil de basses doses (0.2 = ne considère pas les doses < 20% du max de la dose ref)
norm_val = prescription  # Valeur de la dose de normalisation [cGy], ou 'max' pour normaliser par rapport à la dose maximum
doseTA = 5               # Tolérance sur la dose [%]
distTA = 1               # Tolérance sur la distance [mm]
film_filt = 3            # Taille du kernel de filtre médian à appliquer au film pour réduire le bruit.

#%% Préparation
# Initialisation 
film = analysis.DoseAnalysis(film_dose=file_doseFilm, ref_dose=ref_dose,
                             ref_dose_factor=ref_dose_factor, film_dose_factor=film_dose_factor,
                             flipLR=flipLR, flipUD=flipUD, ref_dose_sum=True, rot90=rot90)

film.crop_film()   # Si nécessaire, on peut cropper l'image pour ne garder que le film

# Recalage
film.register(shift_x=shift_x, shift_y=shift_y, threshold=10,
              register_using_gradient=True, markers_center=markers_center)

# Si désiré, on peut utiliser une ROI sur laquelle calcul un facteur de normalisation pour faire correspondre la dose film vs dose ref
film.apply_factor_from_roi()

#%% Écart médian haute dose
thresh = 0.8
ref = prescription
seuil = thresh * ref
medianDiff = film.computeHDmedianDiff(threshold=thresh, ref = ref)
print("Écart médian: {:.2f}% (seuil = {:0.1f} * {} cGy = {} cGy)".format(medianDiff, thresh, ref, seuil))

#%% Effectuer l'analyse gamma
print("Analyse en cours...")
film.gamma_analysis(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)
print("Gammma {}% {}mm: Taux de passage={:.2f}%; Moyenne={:.2f}".format(doseTA, distTA, film.GammaMap.passRate, film.GammaMap.mean))

#%% Afficher et sauvegarder les résultats
film.show_results()
fileout = os.path.join(path_analysis, filebase + ".pdf")
film.publish_pdf(fileout, open_file=True, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)

# On peut sauvegarder les résultats de l'analyse pour pouvoir les charger ultérieurement et afficher les résultats interactifs
# fileOut_pkl = os.path.join(path_analysis, filebase + ".pkl")     
# analysis.save_analysis(film, fileOut_pkl, use_compression=True)
