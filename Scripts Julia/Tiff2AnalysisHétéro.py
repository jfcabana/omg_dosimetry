# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""
import matplotlib
matplotlib.use('TkAgg')

import tiff2dose
import analysis
import os

if __name__ == '__main__':

    # %% Set metadata
    info = dict(author='JMF',
                unit='Salle A',
                film_lot='EBT3XD B2',
                scanner_id='Epson 12000XL',
                date_exposed='2018/11/09',
                date_scanned='2018/11/10',
                wait_time='24 hrs',
                notes=''
                )
    
    # Set filenames and paths
    lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-04-07 Calibration C5/C5_calib_24h_trans_LatCorr_0-500.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-03-25 Calibration lot 04/Calib_C4_24h_trans_LatCorr_0-500.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-03-07 Calibration lot 03/Calib_lot03_24h_trans_LatCorr_0-400.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-02-26 Calibration lot 02/24h/Transmission/Calibration/Calib_24h_trans_LatCorr_0-400.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-02-18 Calibration lot 01/Calib_Electrons_24h_trans_CorrSSD_LatCorr_ProfCorr_Sub3.pkl'

    baseDir = 'P:/Projets/CRIC/Physique_Medicale/Films/20190417 Sein_endtoend'
    film_path = baseDir
    film_filename = 'FactQuot&Jonction001'
    ref_dose = baseDir + '/DoseMonaco/Monaco6MV_62_inch.tif'
    ref_dose_sum = False

    #Scan to dose conversion parameters
    tiff_2_dose = True#True to convert the tiff film file to a dose file, False to not do it
    dose_2_analysis = False #True to perform the gamma analysis, False to not do it
    film_dose_choice = 'Opt' #must be 'Opt', 'RG' or 'Ave'

    #Registration parameters
    QUASAR = False #1 if the film irradiated in the QUASAR phantom, 0 if not
    shift_x = 0.2  # Shift to apply to the film dose in the x direction (mm)
    shift_y = -20  # Shift to apply to the film dose in the y direction (mm)
    threshold_detection = 1  # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.
    flipLR = 1 #1 if the film needs to be flipped in the left-right direction, 0 if not
    flipUD = 1 #1 if the film needs to be flipped in the up-down direction, 0 if not
    rot90 = 1 #1 if the film needs to be turn, 0 if not

    #Dose factors
    ref_dose_factor = 1  # Factor to apply to the reference dose file (for example to convert Gy to cGy)
    film_dose_factor = 0.9791 # DAILY FACTOR  - Factor to apply to the film dose file (for example the daily output factor correction)

    #Gamma analysis parameters
    doseTA1 = 3
    distTA1 = 3
    doseTA2 = 2
    distTA2 = 2
    threshold = 0.1
    norm_val = 'max'
    film_filt = 5
    deleteRegion=False

###########################################################################  AUCUN PARAMETRE À CHANGER SOUS CETTE LIGNE

    #Create paths and filenames
    img_file = os.path.join(film_path, film_filename + '.tif')
    path_dose = os.path.join(film_path, 'DoseFilm')
    if not os.path.exists(path_dose):
        os.makedirs(path_dose)

    path_analysis = os.path.join(baseDir, 'Analyse')
    if not os.path.exists(path_analysis):
        os.makedirs(path_analysis)

    film_dose= os.path.join(path_dose, film_filename[:-4] + '_dose' + film_dose_choice + '.tif')
    analysis_filename = os.path.join(path_analysis, film_filename[:-4])

    #Convert film scan to dose if asked
    if tiff_2_dose:
        # Perform the dose conversion
        gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type='rational', info=info, crop_edges=0)
        # gaf1.show_results()

        # Generate report
        gaf1.publish_pdf(filename=os.path.join(path_dose, film_filename[:-4] + '_Monaco_report.pdf'), open_file=False)

        # %% Save results as tiff images
        # gaf1.dose_r.save(os.path.join(path_dose, film_filename[:-4] + '_doseR.tif'))
        # gaf1.dose_g.save(os.path.join(path_dose, film_filename[:-4] + '_doseG.tif'))
        # gaf1.dose_b.save(os.path.join(path_dose, film_filename[:-4] + '_doseB.tif'))
        gaf1.dose_rg.save(os.path.join(path_dose, film_filename[:-4] + '_doseRG.tif'))
        gaf1.dose_opt.save(os.path.join(path_dose, film_filename[:-4] + '_doseOpt.tif'))
        gaf1.dose_ave.save(os.path.join(path_dose, film_filename[:-4] + '_doseAve.tif'))

    #Perform the analysis
    if dose_2_analysis:
        film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, ref_dose_factor=ref_dose_factor, film_dose_factor=film_dose_factor, ref_dose_sum=ref_dose_sum, deleteRegion=deleteRegion)

        # Perform registration (if needed)
        if QUASAR:
            film.registerQUASAR(shift_x=0, shift_y=0, flipLR=flip, threshold=threshold_detection, displacement_longi=shift_y) #displacement in mm
        else:
            film.register(shift_x=shift_x, shift_y=shift_y, flipLR=flipLR, flipUD=flipUD, rot90=rot90, threshold=threshold_detection)

        # %% Perform analysis and save as pdf for 2 different threshold

        film.analyse(doseTA=doseTA1, distTA=distTA1, threshold=threshold, norm_val=norm_val, film_filt=film_filt, computeIDF= True)
        film.show_results(show=True)
        film.publish_pdf(filename=os.path.join(path_analysis, analysis_filename + str(doseTA1) + '%' +str(distTA1) + 'mm_FQ' + str(film_dose_factor) +'_report.pdf'),
                         open_file=False, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False)

        film.analyse(doseTA=doseTA2, distTA=distTA2, threshold=threshold, norm_val=norm_val, film_filt=film_filt, computeIDF= False)
        film.show_results(show=True)
        film.publish_pdf(filename=os.path.join(path_analysis, analysis_filename + str(doseTA2) + '%' +str(distTA2) + 'mm_FQ' + str(film_dose_factor) +'_report.pdf'),
                         open_file=False, show_hist=False, show_pass_hist=False, show_varDistTA=False, show_var_DoseTA=False)
