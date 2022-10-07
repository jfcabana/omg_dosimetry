#%% Load images and perform registration
import analysis
import os

baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\Films\\QA Patients\\449598\\'
path_in =  baseDir + 'ConvertedDoses\\'
baseName = '385575_ThLSD1A_doseOpt_crop'
path_out = os.path.join(baseDir,'Analysis')
if not os.path.exists(path_out):
    os.makedirs(path_out)

# Path to the film dose image file
film_dose = os.path.join(path_in, baseName + '.tif')

# Path to the reference dose file
ref_dose = baseDir + 'DoseRS\\385575_ThLSD1A_RS.dcm'
film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, film_dose_factor=0.994, ref_dose_factor=1.0, flipLR=False, flipUD=True, ref_dose_sum=False, rot90=0)

#Perform registration (if needed)
shift_x=0   # Shift to apply to the ref dose in the x direction (mm)
shift_y=-0   # Shift to apply to the ref dose in the y direction (mm)
markers_center = [-0.3, -722.7, 211] # Coordinates of the makers intersection in mm (LR, IS, AP), as given in RayStation
threshold=10 # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.
film.register(shift_x=shift_x, shift_y=shift_y, threshold=threshold, register_using_gradient=True, markers_center=markers_center)

#%%
film.apply_factor_from_roi()

#%% Perform analysis

# Gamma analysis parameters
doseTA = 3
distTA = 3
threshold = 0.1
norm_val = 1000
film_filt = 0

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)

film.show_results()
#film.plot_gamma_stats()

# Publish report            
film.publish_pdf(filename=os.path.join(path_out, baseName +'_Ffilm_3%-3mm_report.pdf'), open_file=False, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)