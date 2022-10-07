# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import tiff2dose2
import os

# Set metadata
info = dict(author = 'JFC',
            unit = '3',
            film_lot = 'C8',
            scanner_id = 'Epson 12000XL',
            date_exposed = '2019/10/16',
            date_scanned = '2019/10/23',
            wait_time = '7 jours',
            notes = 'Scan en transmission.'
           )
    

# Set paths
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films'
path_in = path + '\\2019-10-16 Validation Dose Peau\\Scan 2019-10-23\\'
lut_file = path + '\\2019-10-17 Nouvelle calibration C8\\C8_calib_6jours_trans_0-200cGy.pkl'

baseName = '2019-10-16_ValidationDosePeau'
img_file = os.path.join(path_in, baseName + '_001.tif')

fit_type = 'rational'

ext = '_calib0-200_6jours_scale0'

path_out = os.path.join(path_in, 'DoseFilm')
if not os.path.exists(path_out):
    os.makedirs(path_out)
    

# Perform the dose conversion
gaf1 = tiff2dose2.Gaf(path=img_file, lut_file=lut_file, img_filt=0, fit_type=fit_type, info=info, crop_edges=0, clip=300, rot90=0, scale0=True)
#gaf1.show_results()

#%% Generate report
gaf1.publish_pdf(filename=os.path.join(path_out, baseName + ext + '_report.pdf'), open_file=True)

# Save results as tiff images
#gaf1.dose_r.save(os.path.join(path_out,baseName + ext + '_doseR.tif'))
#gaf1.dose_g.save(os.path.join(path_out,baseName + ext + '_doseG.tif'))
#gaf1.dose_b.save(os.path.join(path_out,baseName + ext + '_doseB.tif'))
#gaf1.dose_rg.save(os.path.join(path_out,baseName + ext + '_doseRG.tif'))
#gaf1.dose_ave.save(os.path.join(path_out,baseName + ext + '_doseAve.tif'))
#gaf1.dose_m.save(os.path.join(path_out,baseName + ext + '_doseM.tif'))
##gaf1.dose_opt.crop_edges()
gaf1.dose_opt.save(os.path.join(path_out,baseName + ext + '_doseOpt.tif'))

#%%
