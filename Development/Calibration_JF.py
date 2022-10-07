#%% import librairies
import calibration
import os

#%% Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 3',
            film_lot = 'C7',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2019/09/26 17h',
            date_scanned = '2019/09/27 14h',
            wait_time = '21 heures',
            notes = 'Scan en transmission, avec vitre.'
           )

# Set images folder, define parameters and compute LUT
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-09-26 Calibration C7'
path_in = path + '\\21hr'
outname = 'C7_calib_21h_trans_0-3Gy_vitre'

lateral_correction = 0
#beam_profile = path + '\\ICProfiler 9MeV 25x25 1,9cm.txt'
beam_profile = None

doses = [0, 25, 50, 100, 150, 200, 250, 300]
output = 1

film_detect = True  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size, OR [width, height] (mm). Used only when film_detect is True.
roi_crop = 10        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 0            # Size of the median filter to apply to the film

myLUT = calibration.LUT(path=path_in, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info)

#myLUT.plot_fit(i='mean', fit_type='rational', k=3, ext=3, s=0)

#%% Publish report            
myLUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)

# Save the LUT
filename = os.path.join(path, outname + '.pkl')
calibration.save_lut(myLUT,filename)