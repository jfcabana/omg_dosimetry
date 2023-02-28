#import matplotlib
#matplotlib.use('TkAgg')
#
#import tiff2dose
#import analysis
#import os
#
#if __name__ == '__main__':


#%% Load images and perform registration
from omg_dosimetry import analysis
import os
    
baseName = '123456789_ThLSD1A_doseOpt'    # Nom du fichier de dose à analyser (sans extension .tif)
basePath = 'P:\Projets\CRIC\Physique_Medicale\Films\QA Patients\\123456789\ThLSD_1A'
path_in = os.path.join(basePath, 'DoseFilm')
path_out = os.path.join(basePath, 'Analyse')
path_ref = os.path.join(basePath, 'DoseRS')

if not os.path.exists(path_out):
    os.makedirs(path_out)

# Path to the film dose image file
film_dose = os.path.join(path_in, baseName + '.tif')

# Path to the reference dose file
ref_dose = os.path.join(path_ref, '123456789_ThLSD1A_RS.dcm')

# Prepare analysis
film = analysis.DoseAnalysis(film_dose=film_dose, ref_dose=ref_dose, film_dose_factor=1, flipLR=False, flipUD=True, ref_dose_sum=False, rot90=3)

#Perform registration
shift_x=0   # Shift to apply to the ref dose in the x direction (mm)
shift_y=0   # Shift to apply to the ref dose in the y direction (mm)
markers_center = [-0.3, -722.7, 211] # Coordinates of the makers intersection in mm (LR, IS, AP), as given in RayStation
#markers_center = [0.5, -38.0, 214.5]  # Coordonnées pour Shane
threshold=10 # Threshold parameter for the film edge detection. Increase value if the film is not tightly cropped.


#%% Rouler ce bloc pour normaliser par rapport à une film de normalisation
norm_film_MU = 1000            
norm_film_ref_MU = 1000
norm_film_ref_dose = 973
norm_film_dose = norm_film_MU / norm_film_ref_MU * norm_film_ref_dose
film.apply_factor_from_roi(norm_dose=norm_film_dose)

#%% Rouler ce bloc pour cropper le film
film.crop_film()

#%% Rouler ce bloc pour faire le recalage
film.register(shift_x=shift_x, shift_y=shift_y, threshold=threshold, register_using_gradient=True, markers_center=markers_center)
#%% Rouler ce bloc pour appliquer un facteur de normalisation d'une ROI sur le film
film.apply_factor_from_roi()

#%% Perform analysis 3/3
fileout=os.path.join(path_out, baseName +'_F1_3%-3mm_report.pdf')

# Gamma analysis parameters
doseTA = 3
distTA = 3
threshold = 0.1
norm_val = 1000
film_filt = 3

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)

film.show_results()
#film.plot_gamma_stats()

# Publish report            
film.publish_pdf(fileout, open_file=False, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)

#%% Perform analysis 2/2
fileout=os.path.join(path_out, baseName +'_F1_2%-2mm_report.pdf')

# Gamma analysis parameters
doseTA = 2
distTA = 2
threshold = 0.1
norm_val = 1000
film_filt = 0

film.analyse(doseTA=doseTA, distTA=distTA, threshold=threshold, norm_val=norm_val, film_filt=film_filt)

film.show_results()
#film.plot_gamma_stats()

# Publish report            
film.publish_pdf(fileout, open_file=False, show_hist=True, show_pass_hist=True, show_varDistTA=False, show_var_DoseTA=False, x=None, y=None)