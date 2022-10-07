# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import tiff2dose
import analysis
import os

#def main():
# Set metadata
info = dict(author = 'JFC',
            unit = 'Room A',
            film_lot = 'EBT3 001',
            scanner_id = 'Epson 10000XL',
            date_exposed = '2018/07/15 10am',
            date_scanned = '2018/07/16 10am',
            wait_time = '24 hrs',
            notes = 'Dose analysis demonstration.'
           )

# Set paths
baseDir = 'P:/Projets/CRIC/Physique_Medicale/Films/Hétérogénéité/Crane'
lut_file = 'P:/Projets/CRIC/Physique_Medicale/Films/2019-02-18 Calibration lot 01/Calib 24h/Électrons/Transmission/Calib_Electrons_24h_trans_CorrSSD_LatCorr_ProfCorr_Tous.pkl'

path_in = baseDir
#baseName = 'CalibFilms_All'

baseName = 'Film_20190226_001'
#ext = '_rational_LUTall'

img_file = os.path.join(path_in, baseName + '.tif')
path_out = os.path.join(path_in, 'ConvertedDoses')
if not os.path.exists(path_out):
    os.makedirs(path_out)
    

# =============================================================================
# # Perform the dose conversion
# gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type='rational', info=info, crop_edges=0)
# #gaf1.show_results()
# 
# # Generate report
# gaf1.publish_pdf(filename=os.path.join(path_out, baseName +'_report.pdf'), open_file=True)
# 
# # Save results as tiff images
# #gaf1.dose_m.save(os.path.join(path_out, baseName + '_rational_LUTall_doseM.tif'))
# #gaf1.dose_r.save(os.path.join(path_out, baseName + '_rational_LUTall_doseR.tif'))
# #gaf1.dose_g.save(os.path.join(path_out, baseName + '_rational_LUTall_doseG.tif'))
# #gaf1.dose_b.save(os.path.join(path_out, baseName + '_rational_LUTall_doseB.tif'))
# gaf1.dose_rg.save(os.path.join(path_out, baseName + '_rational_LUTall_doseRG.tif'))
# gaf1.dose_opt.save(os.path.join(path_out,baseName + '_rational_LUTall_doseOpt.tif'))

# gaf1.dose_ave.save(os.path.join(path_out,baseName + '_rational_LUTall_doseAve.tif'))
# 
# =============================================================================
 
 #baseName = 'CalibFilms_All_rational_LUTall_doseRG'
baseName = os.path.join(path_out,baseName + '_rational_LUTall_doseOpt')
path_out = os.path.join(baseDir,'Analysis')
if not os.path.exists(path_out):
    os.makedirs(path_out)
 
 # Path to the film dose image file
film_dose = os.path.join(path_in, baseName + '.tif')
 
 # Path to the reference dose file
 #ref_dose = baseDir + '\\ReferenceDoses\\CalibFilms_All_nominal.tif'
ref_dose = baseDir + '/DoseRS/RD.DirectCrane_6MV.dcm'
film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, film_dose_factor=1)
 
 #Perform registration (if needed)
shift_x=0   # Shift to apply to the film dose in the x direction (mm)
shift_y=0   # Shift to apply to the film dose in the y direction (mm)
threshold=10000 # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.
film.register(shift_x=shift_x, shift_y=shift_y, flipLR=False, threshold=threshold)
      
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
