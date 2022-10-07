# -*- coding: utf-8 -*-
"""
This file provides a demonstration of the calibration module
"""
#%% import librairies
import calibration
import os

#%% Set metadata
info = dict(author = 'JMF',
            unit = 'Salle A',
            film_lot = 'EBT3 XD B2',
            scanner_id = 'Epson 10000XL',
            date_exposed = '2019/01/08 16h30',
            date_scanned = '2019/01/09 16h30',
            wait_time = '24 hrs',
            notes = 'sans corrections latérales, courbe rational'
           )

#%% Set images folder, define parameters and compute LUT
path ='R:/07 Projet/2018 SBRT/Films gafchromiques/Calibration/2019-01-08/10Films/'
output_name = 'calibration_withLC_rational_2019-01-08'

lateral_correction=True
beam_profile = None
doses =  [0, 79.07,	158.15,	256.99,	425.03,	612.83,	840.17,	1097.16, 1383.80, 1680.33] #in cGy
output = 1.0

film_detect = False  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size, OR [width, height] (mm). Used only when film_detect is True.
roi_crop_width = 8        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
roi_crop_length = 8        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 5            # Size of the median filter to apply to the film
fit_type = 'rational' # must be rational or spline

####################################################################################

myLUT = calibration.LUT(path=path, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop_width=roi_crop_width, roi_crop_length=roi_crop_length, filt=filt, info=info)


#%% Publish report            
myLUT.publish_pdf(filename=os.path.join(path, output_name +'_report.pdf'), open_file=True, fit_type=fit_type)

#%% Save the LUT
filename = os.path.join(path, output_name + '.pkl')
calibration.save_lut(myLUT,filename)
myLUT.save_calibration_curves(outputfile=os.path.join(path, output_name + '.txt'))
