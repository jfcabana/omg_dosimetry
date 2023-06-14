# -*- coding: utf-8 -*-
"""
This script is used to demonstrate an example using the calibration module of omg_dosimetry.
You can make a copy and adapt it according to your needs.
    
Écrit par Jean-François Cabana
jean-francois.cabana.cisssca@ssss.gouv.qc.ca
2023-03-01
"""

#%% Import libraries
from omg_dosimetry import calibration
import os

#%% Define general information
info = dict(author = 'Demo Physicist',
            unit = 'Demo Linac',
            film_lot = 'XD_1',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-01-24 16h',
            date_scanned = '2023-01-25 16h',
            wait_time = '24 hours',
            notes = 'Transmission mode @300ppp'
           )

path = os.path.join(os.path.dirname(__file__), "files", "calibration") ## Root folder
path_scan = os.path.join(path, "scan")                                 ## Folder containing scanned images
outname = 'Demo_calib'                                                 ## Name of the calibration file to produce

#%% Set calibration parameters
#### Dose
doses = [0.0, 100.0, 200.0, 400.0, 650.0, 950.0]      ## Imparted doses [cGy] to the films
output = 1.0                                          ## If necessary, correction for the daily output of the machine

### Lateral correction
lateral_correction = True                             ## True to perform a calibration with lateral correction of the scanner (requires long strips of film)
                                                         # or False for calibration without lateral correction
beam_profile = os.path.join(path, "BeamProfile.txt")  ## None to not correct for the shape of the dose profile,
                                                         # or path to a text file containing the shape profile

### Film detection
film_detect = True      ## True to attempt automatic film detection, or False to make a manual selection
crop_top_bottom = 650   ## If film_detect = True: Number of pixels to crop in the top and bottom of the image.
                           # May be required for auto-detection if the glass on the scanner is preventing detection
roi_size = 'auto'       ## If film_detect = True: 'auto' to define the size of the ROIs according to the films,
                           # or [width, height] (mm) to define a fixed size.
roi_crop = 3            ## If film_detect = True and roi_size = 'auto': Margin size [mm] to apply on each side
                           # films to define the ROI.

### Image filtering
filt = 3                ## Median filter kernel size to apply on images for noise reduction

#%% Produce the LUT
LUT = calibration.LUT(path=path_scan, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info, crop_top_bottom = crop_top_bottom)

#%% View results and save LUT
LUT.plot_roi()  # To display films and ROIs used for calibration
LUT.plot_fit()  # To display a plot of the calibration curve and the fitted algebraic function
LUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)            # Publication of the PDF report
calibration.save_lut(LUT, filename=os.path.join(path, outname + '.pkl'), use_compression=True)  # Saving the LUT file. use_compression allows a reduction  
                                                                                                # in file size by a factor of ~10, but slows down the operation.