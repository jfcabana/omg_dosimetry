#%% Load images and perform registration
import analysis
import os
import numpy as np
import pickle
import imageRGB

baseName = 'B1A5'
ext = '_spline_opt.tif'

#baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Champs complexes\\Champs complexes 2\\'
baseDir = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Plans complexes\\'
sub_dir = baseDir + 'VersaHD_SRS1\\'

path_in =  baseDir + 'Dose_spline\\'
path_out = os.path.join(sub_dir,'Analysis')
if not os.path.exists(path_out):
    os.makedirs(path_out)

# Path to the film dose image file
film_dose = os.path.join(path_in, baseName + ext)

# Path to the reference dose file
ref_dose = sub_dir + 'DoseRS\\' + baseName + '.dcm'
#ref_dose = sub_dir + 'DoseRS\\' + baseName[:-1] + '1.dcm'


film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, flipLR=False, flipUD=True, ref_dose_sum=False, rot90=-1)

#Perform registration (if needed)
shift_x=0   # Shift to apply to the ref dose in the x direction (mm)
shift_y=-0   # Shift to apply to the ref dose in the y direction (mm)
markers_center = [-0.3, -722.7, 211] # Coordinates of the makers intersection in mm (LR, IS, AP), as given in RayStation
threshold=10 # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.
film.register(shift_x=shift_x, shift_y=shift_y, threshold=threshold, register_using_gradient=True, markers_center=markers_center)

#%%
#film.apply_factor_from_roi()


#%% Perform analysis

# Gamma analysis parameters
doseTA = 3
distTA = 1
threshold = 0.4
#norm_val = 'max'
norm_val = film.ref_dose.array.max() * 0.7
film_filt = 0

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)

film.show_results()
#film.plot_gamma_stats()

# Publish report         
fileOut_pdf = os.path.join(path_out, baseName + '_' + str(doseTA) + '%-' + str(distTA) + 'mm.pdf')   
film.publish_pdf(fileOut_pdf, open_file=False, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)

# Get median diff
film.computeHDmedianDiff(threshold=0.7)
print("Écart médian haute dose: " + str(film.HD_median_diff) + "%")
#HDthreshold = norm_val
#film_HD = film.film_dose.array[film.ref_dose.array > HDthreshold]
#ref_HD = film.ref_dose.array[film.ref_dose.array > HDthreshold]
#HD_median_diff = np.median((film_HD-ref_HD)/ref_HD) * 100

# write to pickle
fileOut_pkl = os.path.join(path_out, baseName + '_' + str(doseTA) + '%-' + str(distTA) + 'mm.pkl')   
with open(fileOut_pkl, 'wb') as fp:
	pickle.dump(film, fp)


#%% Gamma analysis parameters
doseTA = 3
distTA = 1
threshold = 0.4
#norm_val = 'max'
norm_val = film.ref_dose.array.max() * 0.7
film_filt = 0

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)
film.show_results()

#%% Load pickle and view results
fileIn_pkl = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Champs complexes\\Champs complexes 1\\VersaHD_SRS\\Analysis\\C1A10_3%-1mm.pkl'
#fileIn_pkl = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Plans complexes\\LF0TG0.4\\Analysis\\B1A_3%-1mm.pkl'

with open (fileIn_pkl, 'rb') as fp:
    film = pickle.load(fp)
film.show_results()

#%% ****** pour l'instant ça marche pas ***** Load pickle, change ref dose and perform new analysis 
fileIn_pkl = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Champs complexes\\Champs complexes 2\\T&G 0,3mm\\Analysis\\C1B9_10_3%-1mm.pkl'
new_ref_dose = 'P:\\Projets\\CRIC\\Physique_Medicale\\SRS\\Commissioning\\Champs complexes\\Champs complexes 2\\T&G 0,3mm\\DoseRS\\C1B9_10_.dcm'

with open (fileIn_pkl, 'rb') as fp:
    film = pickle.load(fp)
    
film.ref_dose = imageRGB.load(ref_dose)

(film.film_dose, film.ref_dose) = imageRGB.equate_images(film.film_dose, film.ref_dose)
film.ref_dose.path = new_ref_dose   
# Gamma analysis parameters
doseTA = 3
distTA = 1
threshold = 0.4
#norm_val = 'max'
norm_val = film.ref_dose.array.max() * 0.7
film_filt = 0

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)

film.show_results()