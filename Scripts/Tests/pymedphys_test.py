9# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import matplotlib.pyplot as plt
import imageRGB
import pymedphys
# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

path = "P:\\Projets\\CRIC\\Physique_Medicale\\Films\\QA Patients\\0phy_SRS_multi\\A1A_1x1_2cm\\"
path_dose_film = path + "DoseFilm.tiff"
# path_dose_film = path + "DoseRS_noise.tiff"
path_dose_ref = path + "DoseRS.tiff"
film_dose = imageRGB.load(path_dose_film)
ref_dose = imageRGB.load(path_dose_ref)

dose_percent_threshold = 1
distance_mm_threshold = 1

# set coordinates [mm]
x_coord = (np.array(range(0, ref_dose.shape[0])) / film_dose.dpmm - ref_dose.physical_shape[0]/2).tolist()
y_coord = (np.array(range(0, ref_dose.shape[1])) / film_dose.dpmm - ref_dose.physical_shape[1]/2).tolist()
axes_reference = (x_coord, y_coord)
axes_evaluation = (x_coord, y_coord)
dose_reference = ref_dose.array
dose_evaluation = film_dose.array

gamma = pymedphys.gamma(
    axes_reference, dose_reference, 
    axes_evaluation, dose_evaluation, 
    dose_percent_threshold, distance_mm_threshold)

valid_gamma = gamma[~np.isnan(gamma)]

out = plt.hist(valid_gamma, bins=20, density=True)
plt.xlabel("gamma index")
_ = plt.ylabel("probability density")
print(f"Pass rate(\u03B3<=1): {len(valid_gamma[valid_gamma <= 1]) / len(valid_gamma) * 100}%")

# gamma_options = {
#     'dose_percent_threshold': 1,
#     'distance_mm_threshold': 1,
#     'lower_percent_dose_cutoff': 20,
#     'interp_fraction': 10,  # Should be 10 or more for more accurate results
#     'max_gamma': 2,
#     'random_subset': None,
#     'local_gamma': True,
#     'ram_available': 2**29  # 1/2 GB
# }
    
# gamma = pymedphys.gamma(
#     axes_reference, dose_reference, 
#     axes_evaluation, dose_evaluation, 
#     **gamma_options)

# valid_gamma = gamma[~np.isnan(gamma)]

# num_bins = (
#     gamma_options['interp_fraction'] * gamma_options['max_gamma'])
# bins = np.linspace(0, gamma_options['max_gamma'], num_bins + 1)

# plt.hist(valid_gamma, bins, density=True)
# #if density is True, y value is probability density; otherwise, it is count in a bin
# plt.xlim([0, gamma_options['max_gamma']])
# plt.xlabel('gamma index')
# plt.ylabel('probability density')
    
# pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)