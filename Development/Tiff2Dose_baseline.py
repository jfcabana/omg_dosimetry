# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import tiff2dose
import os

#%% Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 2',
            film_lot = 'CRIC 02',
            scanner_id = 'Epson 12000XL',
            date_exposed = '2019/03/02 8:00',
            date_scanned = '2019/03/04 8:00',
            wait_time = '48 hrs',
            notes = 'Scan en réflexion.'
           )

# Set paths

baseDir = path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-03-07 Calibration lot 03\\'

baseline = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Baseline scanner\\Transmission\\Transmission_001.tif'
lut_file =  baseDir + 'Calib_lot03_24h_trans_LatCorr_0-400_baseline.pkl'

path_in = baseDir + 'PDD_trans'

baseName = 'Calib_03_refl'
img_file = os.path.join(path_in, baseName + '_001.tif')

fit_type = 'rational'

ext = ''

path_out = os.path.join(path_in, 'ConvertedDoses')
if not os.path.exists(path_out):
    os.makedirs(path_out)
    

# Perform the dose conversion
gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, fit_type=fit_type, info=info, crop_edges=0, clip=900, rot90=False, baseline=baseline)
#gaf1.show_results()

# Generate report
# gaf1.publish_pdf(filename=os.path.join(path_out, baseName + ext + '_report.pdf'), open_file=True)

# Save results as tiff images
#gaf1.dose_r.save(os.path.join(path_out,baseName + ext + '_doseR.tif'))
#gaf1.dose_g.save(os.path.join(path_out,baseName + ext + '_doseG.tif'))
#gaf1.dose_b.save(os.path.join(path_out,baseName + ext + '_doseB.tif'))
gaf1.dose_opt.crop_edges()
gaf1.dose_opt.save(os.path.join(path_out,baseName + ext + '_doseOpt.tif'))