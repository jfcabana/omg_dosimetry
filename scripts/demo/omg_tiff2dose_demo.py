# -*- coding: utf-8 -*-
"""
This script is used to demonstrate an example using the tiff2dose module of omg_dosimetry.
You can make a copy and adapt it according to your needs.
    
Written by Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Import libraries
from omg_dosimetry import tiff2dose
import os

#%% Define general information
info = dict(author = 'Demo Physicist',
            unit = 'Demo Linac',
            film_lot = 'XD_1',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-01-24 16h',
            date_scanned = '2023-01-25 16h',
            wait_time = '24 hours',
            notes = 'Transmission mode, @300ppp and 16 bits/channel'
           )

path = os.path.join(os.path.dirname(__file__), "files", "tiff2dose")   ## Root folder
path_scan = os.path.join(path, "scan",'A1A_Multi_6cm_001.tif')         # ## Folder containing scanned images, or file path if folder contains multiple film scans.
outname = "Demo_dose"                                                  ## Name of the output file to produce

#%% Dose conversion parameters
lut_file = os.path.join(os.path.dirname(__file__), "files", "calibration","Demo_calib.pkl")   # Path to LUT film to use
fit_type = 'rational'                                                                         # Function type used for fitting calibration curve. 'rational' (recommended) or 'spline'
clip = 500                                                                                    # Maximum value [cGy] to limit dose. Useful to avoid very high doses obtained due to markings on the film.

#%% Perform dose conversion
gaf1 = tiff2dose.Gaf(path=path_scan, lut_file=lut_file, fit_type=fit_type, info=info, clip = clip)

#%% Save dose and PDF report
filename_tif = os.path.join(path, outname+'.tif')
gaf1.dose_opt.save(filename_tif)                    # We save the optimized dose (dose_opt). Other options include individual channels (dose_r, dose_g, dose_b) and individual channels doses average (dose_ave).

filename_pdf = os.path.join(path, outname+'.pdf')
gaf1.publish_pdf(filename_pdf, open_file=True)