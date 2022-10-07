# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the Tiff2Dose module
"""

import tiff2dose
import os

# Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 2',
            film_lot = 'C10',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2020-07-27 16h',
            date_scanned = '2020-07-28 16h',
            wait_time = '24 heures',
            notes = 'Scan en transmission avec la vitre.'
           )

# Set paths
#path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2020-07-28 Dose Peau'
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibration C10\\Valid 24h'
path_in = path
lut_file = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibration C10\\C10_calib_24h_trans_vitre_0-2Gy.pkl'

baseName = 'C10_valid24h_part2'
img_file = os.path.join(path_in, baseName + '_001.tif')

fit_type = 'spline'

ext = '_0-2Gy' + fit_type

path_out = os.path.join(path_in, 'DoseFilm')
if not os.path.exists(path_out):
    os.makedirs(path_out)
    

# Perform the dose conversion
gaf1 = tiff2dose.Gaf(path=img_file, lut_file=lut_file, img_filt=0, fit_type=fit_type, info=info, crop_edges=0, clip=200, rot90=0, scale0=0)
#gaf1.show_results()

# Generate report
gaf1.publish_pdf(filename=os.path.join(path_out, baseName + ext + '_report.pdf'), open_file=1)

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
