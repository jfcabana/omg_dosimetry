# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the analysis module
"""
#%% Load images and perform registration
import analysis
import os

baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-02-18 Calibration lot 01\\Calib 24h\\Validation\\transmission\\PDD'
path_in =  baseDir + '\ConvertedDoses'
#baseName = 'CalibFilms_All_rational_LUTall_doseRG'
baseName = 'PDD10x10_6MV_2019-02-22_trans_Sub3_rational_LatCorr_doseOpt'
path_out = os.path.join(baseDir,'Analysis')
if not os.path.exists(path_out):
    os.makedirs(path_out)

# Path to the film dose image file
film_dose = os.path.join(path_in, baseName + '.tif')

# Path to the reference dose file
#ref_dose = baseDir + '\\ReferenceDoses\\CalibFilms_All_nominal.tif'
#ref_dose = baseDir + '\\PDD10x10_6MV_RSdose_sag.dcm'
#ref_dose = baseDir + '\\RSdose\\'

ref_dose = 'H:\\ExportRS\\VMAT_SEIN 0DOSI DOSI_STB_SEING\\SEIN_1A\\'
film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, film_dose_factor=1.00, ref_dose_factor=1, flipLR=False, ref_dose_sum=True)

#Perform registration (if needed)
shift_x=-150   # Shift to apply to the ref dose in the x direction (mm)
shift_y=0   # Shift to apply to the ref dose in the y direction (mm)
threshold=10000 # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.
film.register(shift_x=shift_x, shift_y=shift_y, threshold=threshold)

#%% Perform analysis

# Gamma analysis parameters
doseTA = 2
distTA = 2
threshold = 0.1
norm_val = 'max'
film_filt = 5

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)

film.show_results()
#film.plot_gamma_stats()

# Publish report            
film.publish_pdf(filename=os.path.join(path_out, baseName +'_report.pdf'), open_file=True, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False)