#%% import librairies
import calibration
import os

# Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 4',
            film_lot = 'C9-XD',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2019/11/12 17h',
            date_scanned = '2019/11/13 11h',
            wait_time = '18 heures',
            notes = 'Scan en transmission avec la vitre.'
           )

# Set images folder, define parameters and compute LUT
path = '.\\DemoTransmissionNoLatCorr\\Calibration'
outname = 'DemoCalibrationTransmissionNoLatCorr_0-28Gy'

doses = [0.00, 78.70, 236.10, 472.20, 787.00, 1180.50, 1653.49, 2204.39, 2833.99]

output = 1

film_detect = False  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size, OR [width, height] (mm). Used only when film_detect is True.
roi_crop = 5        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 0            # Size of the median filter to apply to the film

myLUT = calibration.LUT(path=path, doses=doses, output=output, lateral_correction=False, beam_profile=None,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info)

#myLUT.plot_fit(i='mean', fit_type='rational', k=3, ext=3, s=0)

#%% Publish report            
myLUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)

# Save the LUT
filename = os.path.join(path, outname + '.pkl')
calibration.save_lut(myLUT,filename)