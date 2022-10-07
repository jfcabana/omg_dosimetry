# -*- coding: utf-8 -*-
"""
This file analyse a film irradiated by a 10x10 field to compute a daily factor.
"""
import matplotlib
matplotlib.use('TkAgg')

import tiff2dose
import analysis
import os

if __name__ == '__main__':

    # Set filenames and paths
    lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-04-07 Calibration C6 XD/C6-XD_calib_24h_trans_LatCorr_0-24Gy_vitre.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-04-07 Calibration C5/C5_calib_24h_trans_LatCorr_0-500_vitre.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-03-25 Calibration lot 04/Calib_C4_96h_trans_LatCorr_0-500.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-03-07 Calibration lot 03/Calib_lot03_24h_trans_LatCorr_0-400.pkl'
    #lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-02-26 Calibration lot 02/24h/Transmission/Calibration/Calib_24h_trans_LatCorr_0-400.pkl'
    baseDir = 'P:/Projets/CRIC/Physique_Medicale/Films/QA Patients/478371+486918'
    film_path = baseDir
    film_filename = 'FilmC6_20190621_001'
    ref_dose = 'P:/Projets/CRIC/Physique_Medicale/Films/QA Patients/10x10/DoseRS/RD.Classique_10x10_6MV_100MU.dcm'
    #ref_dose = 'P:/Projets/CRIC/Physique_Medicale/Films/QA Patients/10x10/DoseRS/RD.10x10_6MV_300MU.dcm'

    #Scan to dose conversion parameters
    tiff_2_dose = False #1 to convert the tiff film file to a dose file, 0 to not to it
    film_dose_choice = 'Opt' #must be 'Opt', 'RG' or 'Ave'

    #Registration parameters
    markers_center = [-0.3, -722.7, 211]  # Coordinates of the makers intersection in mm (LR, IS, AP), as given in RayStation
    QUASAR = False #1 if the film irradiated in the QUASAR phantom, 0 if not
    #shift_x = -96.7  # Shift to apply to the film dose in the x direction (mm)
    #shift_y = 25.7  # Shift to apply to the film dose in the y direction (mm)
    threshold_detection = 1  # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.
    rot90 = 1 #1 if the film needs to be turn, 0 if not
    flipUD = 1
    flipLR = 0
    #Dose factors
    ref_dose_factor = 100  # Factor to apply to the reference dose file (for example to convert Gy to cGy)
    MU = 800 # monitor units for the 10x10 field

####################################################################################

    #Create paths and filenames
    img_file = os.path.join(film_path, film_filename + '.tif')
    path_dose = os.path.join(film_path, 'DoseFilm')
    if not os.path.exists(path_dose):
        os.makedirs(path_dose)

    film_dose= os.path.join(path_dose, film_filename[:-4] + '_dose' + film_dose_choice + '.tif')

    #Convert film scan to dose if asked
    if tiff_2_dose:
        # Perform the dose conversion
        gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type='rational', crop_edges=1)
        # gaf1.show_results()

        # Generate report
        gaf1.publish_pdf(filename=os.path.join(path_dose, film_filename[:-4] + '_report.pdf'), open_file=True)

        # %% Save results as tiff images
        # gaf1.dose_r.save(os.path.join(path_dose, film_filename + '_doseR.tif'))
        # gaf1.dose_g.save(os.path.join(path_dose, film_filename + '_doseG.tif'))
        # gaf1.dose_b.save(os.path.join(path_dose, film_filename + '_doseB.tif'))
        gaf1.dose_rg.save(os.path.join(path_dose, film_filename[:-4] + '_doseRG.tif'))
        gaf1.dose_opt.save(os.path.join(path_dose, film_filename[:-4] + '_doseOpt.tif'))
        gaf1.dose_ave.save(os.path.join(path_dose, film_filename[:-4] + '_doseAve.tif'))

    film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, ref_dose_factor=ref_dose_factor*MU/100)

    # Perform registration (if needed)
    if QUASAR:
        film.registerQUASAR(shift_x=0, shift_y=0, flipLR=True, rot90=rot90, threshold=threshold, displacement_longi=shift_y) #displacement in mm
    else:
        film.register(flipLR=flipLR, flipUD=flipUD, rot90=rot90, threshold=threshold_detection, markers_center=markers_center)

    # %% Perform analysis and save as pdf for 2 different threshold
    
    film.computeDailyFactor()
