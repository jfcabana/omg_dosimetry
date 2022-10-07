#%% import librairies
import calibration
import os
import matplotlib.pyplot as plt

# Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 2',
            film_lot = 'C2',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2019/03/07',
            date_scanned = '2019/03/08',
            wait_time = '24 heures',
            notes = 'Scan en réflexion.'
           )

# Set images folder, define parameters and compute LUT
path = '.\\DemoReflexionLatCorr\\Calibration'
outname = 'DemoCalibrationReflexionLatCorr_0-8Gy'

lateral_correction = True
beam_profile = path + '\\ICProfiler 9MeV 25x25 1,9cm.txt'

doses = [0., 25., 50., 100., 150., 200., 250., 300., 400., 800.]


output = 1

film_detect = True  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size, OR [width, height] (mm). Used only when film_detect is True.
roi_crop = 3        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 0            # Size of the median filter to apply to the film

myLUT = calibration.LUT(path=path, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info)

#myLUT.plot_fit(i='mean', fit_type='rational', k=3, ext=3, s=0)

#%% Publish report            
myLUT.plot_roi()
myLUT.plot_beam_profile()
myLUT.plot_calibration_curves(mode='all')
ax=plt.gca()
myLUT.plot_fit(ax=ax)
myLUT.plot_lateral_response()
myLUT.plot_profile()

myLUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)

# Save the LUT
filename = os.path.join(path, outname + '.pkl')
calibration.save_lut(myLUT,filename)