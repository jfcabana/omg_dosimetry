# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import tiff2dose
import os

# Set metadata
info = dict(author = 'JFC',
            unit = 'Room A',
            film_lot = 'C3',
            scanner_id = 'Epson 10000XL',
            date_exposed = '2019/03/07',
            date_scanned = '2019/03/08',
            wait_time = '24 hrs',
            notes = 'Tiff2Dose Demonstration.'
           )

# Set paths
baseDir = '.\\DemoReflexionLatCorr'
lut_file = baseDir + '\\Calibration\\DemoCalibrationReflexionLatCorr_0-4Gy.pkl'

path_in = baseDir + '\\Scan'
#baseName = 'CalibFilms_All'

baseName = 'PDD10x10_6MV_2019-02-22_refl_001'
ext = '_demo'

img_file = os.path.join(path_in, baseName + '.tif')
path_out = os.path.join(baseDir, 'ConvertedDoses')
if not os.path.exists(path_out):
    os.makedirs(path_out)

# Perform the dose conversion
gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, info=info)
#gaf1.show_results()

# Generate report
gaf1.publish_pdf(filename=os.path.join(path_out, baseName +'_report.pdf'), open_file=True)

#%% Save results as tiff images
#gaf1.dose_m.save(os.path.join(path_out, baseName + '_doseM.tif'))
#gaf1.dose_r.save(os.path.join(path_out, baseName + '_doseR.tif'))
#gaf1.dose_g.save(os.path.join(path_out, baseName + '_doseG.tif'))
#gaf1.dose_b.save(os.path.join(path_out, baseName + '_doseB.tif'))
#gaf1.dose_rg.save(os.path.join(path_out, baseName + '_doseRG.tif'))
gaf1.dose_opt.save(os.path.join(path_out,baseName + '_doseOpt.tif'))
#gaf1.dose_ave.save(os.path.join(path_out,baseName + '_doseAve.tif'))