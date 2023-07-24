# -*- coding: utf-8 -*-
"""
This script is used to demonstrate an example using the analysis module of omg_dosimetry.
You can make a copy and adapt it according to your needs.
    
Written by Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Import libraries
from omg_dosimetry import analysis
import os
    
#%% Define file paths
path_analysis = os.path.join(os.path.dirname(__file__), "files", "analysis")                   # Root folder
ref_dose = os.path.join(path_analysis, 'DoseRS')                                               # Folder containing reference planar dose
file_doseFilm = os.path.join(os.path.dirname(__file__), "files", "tiff2dose", "Demo_dose.tif") # Path to converted film dose (must have run omg_tiff2dose_demo first)
filebase = 'Demo_analysis'                                                                     # File name of the output PDF report

#%% Define analysis parameters
#### Normalisation factors
ref_dose_factor = 1.0   # Normalisation factor to apply to reference dose
film_dose_factor = 1.0  # Normalisation factor to apply to film dose
prescription = 300      # Prescription dose [cGy]

#### Paramètres de recalage 
flipLR = True           # True/False: apply an horizontal mirror transformation to film dose
flipUD = False          # True/False: apply a vertical mirror transformation to film dose
rot90 = 1               # int: numver of 90 degrees rotation to apply to film dose
shift_x = 0.0           # float [mm]: If necessary, apply a known shift to the film in the X direction
shift_y = -0.8          # float [mm]: If necessary, apply a known shift to the film in the Y direction
markers_center = [0.8, 1.2, 233.3]  # Coordinates [mm] in the reference dose corresponding to the marks intersection on the film (R-L, I-S, P-A)

#### Gamma analysis parameters
threshold = 0.20         # Low dose threshold (e.g. 0.2: don't consider dose < 20% of normalisation dose)
norm_val = prescription  # Normalisation dose [cGy], or 'max' to normalise with respect to reference dose maximum
doseTA = 5               # Dose to agreement threshold [%]
distTA = 1               # Distance to agreement threshold [mm]
film_filt = 3            # Size of median filter kernel to apply to film dose for noise reduction

#%% Preparation
# Initialisation 
film = analysis.DoseAnalysis(film_dose=file_doseFilm, ref_dose=ref_dose,
                             ref_dose_factor=ref_dose_factor, film_dose_factor=film_dose_factor,
                             flipLR=flipLR, flipUD=flipUD, ref_dose_sum=True, rot90=rot90)

# If necessary, we can crop the film region before analysis
film.crop_film()

# Perform registration
film.register(shift_x=shift_x, shift_y=shift_y, threshold=10,
              register_using_gradient=True, markers_center=markers_center)

# We can define an ROI on the film to compute a normalisation factor to match to reference dose
film.apply_factor_from_roi()

#%% Get high dose deviation
thresh = 0.8
ref = prescription
seuil = thresh * ref
medianDiff = film.computeHDmedianDiff(threshold=thresh, ref = ref)
print("Écart médian: {:.2f}% (seuil = {:0.1f} * {} cGy = {} cGy)".format(medianDiff, thresh, ref, seuil))

#%% Perform gamma analysis
print("Analyse en cours...")
film.gamma_analysis(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)
print("Gammma {}% {}mm: Taux de passage={:.2f}%; Moyenne={:.2f}".format(doseTA, distTA, film.GammaMap.passRate, film.GammaMap.mean))

#%% Show results and save report
film.show_results()
fileout = os.path.join(path_analysis, filebase + ".pdf")
film.publish_pdf(fileout, open_file=True, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)

# Results can be saved to file so they can be loaded back later to access interactive visualisation
# fileOut_pkl = os.path.join(path_analysis, filebase + ".pkl")     
# analysis.save_analysis(film, fileOut_pkl, use_compression=True)
