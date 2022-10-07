#%% import librairies
import calibration
import os

#%% Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 3',
            film_lot = 'C13 EBT3',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2022-02-03 17h',
            date_scanned = '2022-02-04 11h',
            wait_time = '18 heures',
            notes = 'Scan en transmission @300ppp'
           )

# Set images folder, define parameters and compute LUT
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2022-02-03 Calibration C13\\0-3Gy 18h'
path_in = path + '\\Scan'
outname = 'C13_calib_18h_trans_300ppp_0-3Gy'

lateral_correction = 0
#beam_profile = path + '\\ICProfiler 9MeV 25x25 1,9cm.txt'
beam_profile = None

#doses = [0.00, 15.74, 31.48, 55.09, 90.51, 129.86, 173.14, 220.36, 275.45, 338.41, 417.11, 511.55, 629.60, 787.00, 983.75]
doses = [0.00, 15.74, 31.48, 55.09, 90.51, 129.86, 173.14, 220.36, 275.45, 338.41]

output = 1.0

film_detect = True  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size, OR [width, height] (mm). Used only when film_detect is True.
roi_crop = 6        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 0            # Size of the median filter to apply to the film

myLUT = calibration.LUT(path=path_in, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info)

#myLUT.plot_fit(i='mean', fit_type='rational', k=3, ext=3, s=0)

#%% Publish report            
myLUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)

# Save the LUT
filename = os.path.join(path, outname + '.pkl')
calibration.save_lut(myLUT,filename)