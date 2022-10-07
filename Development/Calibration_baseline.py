#%% import librairies
import calibration
import os

# Set metadata
info = dict(author = 'JFC',
            unit = 'Salle 2',
            film_lot = 'CRIC 03',
            scanner_id = 'Epson 12000XL',
            date_exposed = '2019/03/07 11:30',
            date_scanned = '2019/03/08 11:30',
            wait_time = '24 hrs',
            notes = 'Scan en transmission.'
           )

# Set images folder, define parameters and compute LUT
path = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-03-07 Calibration lot 03\\Calib_lat_trans'
outname = 'Calib_lot03_24h_trans_LatCorr_0-400_baseline'

#baseline = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Baseline scanner\\Baseline_trans.tif'
baseline = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\Baseline scanner\\Transmission\\Transmission_001.tif'

lateral_correction=True
beam_profile = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\2019-03-07 Calibration lot 03\\ICProfiler 9MeV 25x25 1,9cm.txt'
#beam_profile = None

doses = [0, 24.86, 49.73, 99.45, 149.18, 198.90, 248.63, 298.35, 348.08, 397.80]
#doses = [0, 24.86, 49.73, 99.45, 149.18, 198.90, 248.63, 298.35]
#doses = [0, 25, 50, 100, 150, 200, 250, 300, 350, 400]
output = 1.0

film_detect = False  # Performs automatic film detection
roi_size = 'auto'   # Either 'auto' to have ROIs defined by films size, OR [width, height] (mm). Used only when film_detect is True.
roi_crop = 2        # Size (mm) of the margin to apply to film strip to crop the ROIs. Used only when film_detect is True and roi_size is 'auto'
filt = 5            # Size of the median filter to apply to the film

myLUT = calibration.LUT(path=path, doses=doses, output=output, lateral_correction=lateral_correction, beam_profile=beam_profile,
                        film_detect=film_detect, roi_size=roi_size, roi_crop=roi_crop, filt=filt, info=info, baseline = baseline)

#myLUT.plot_fit(i='mean', fit_type='rational', k=3, ext=3, s=0)
#%% Publish report            
myLUT.publish_pdf(filename=os.path.join(path, outname +'_report.pdf'), open_file=True)

# Save the LUT
filename = os.path.join(path, outname + '.pkl')
calibration.save_lut(myLUT,filename)