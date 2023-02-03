#%% import librairies
import calibration
import os

#%% Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 2',
            film_lot = 'C14 XD',
            scanner_id = 'Epson 72000XL',
            date_exposed = '2023-01-24 16h',
            date_scanned = '2023-01-25 10h',
            wait_time = '48 heures',
            notes = 'Scan en transmission @300ppp'
           )

# Set images folder. define parameters and compute LUT
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibrations\\2023-01-23 C14 XD\\18h\\Test'
path_in = path + '\\Scan'
# outname = 'C14_calib_48h_trans_300ppp_0-30Gy_LatCorr_BeamCorr'
outname = 'test'

lateral_correction = True
beam_profile = None
beam_profile = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Calibrations\\2023-01-23 C14 XD\\BeamProfile.txt"

doses = [0.00, 100.00, 200.00, 400.00, 650.00, 950.00, 1300.00, 1700.00, 2150.00, 2650.00, 3200.00]
#doses = [0.00, 100.00, 200.00, 400.00, 650.00, 950.00]

output = 1.0
crop_top_bottom = 650
#crop_top_bottom = 1500
film_detect = True  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size. OR [width. height] (mm). Used only when film_detect is True.
roi_crop = 3        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 3            # Size of the median filter to apply to the film

LUT = calibration.LUT(path=path_in, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info, crop_top_bottom = crop_top_bottom)
#LUT.plot_roi()
#LUT.plot_fit(i='mean', fit_type='rational', k=3, ext=3, s=0)

#%% Publish report            
LUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)

# Save the LUT
filename = os.path.join(path, outname + '.pkl')
calibration.save_lut(LUT, filename)