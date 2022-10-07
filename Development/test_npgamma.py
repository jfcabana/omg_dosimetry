# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:26:43 2018

@author: caje1277
"""

#%%
import imageRGB
from npgamma import calc_gamma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time



path = 'S:\\Python\\Film Dosimetry\\Tests\\FilmsJulia\\'
film_dose_file = path + 'film_dose_reg.tiff'
ref_dose_file = path + 'ref_dose_reg.tiff'

film_dose = imageRGB.load(film_dose_file)
ref_dose = imageRGB.load(ref_dose_file)

doseTA = 2
distTA = 2
threshold = 0.1
norm_val = 'max'
film_filt = 3


if norm_val is not None:
    if norm_val is 'max':
        norm_val = ref_dose.array.max()
    film_dose.normalize(norm_val)
    ref_dose.normalize(norm_val)

## invalidate dose values below threshold so gamma doesn't calculate over it
#ref_dose.array[ref_dose.array < threshold * np.max(ref_dose)] = np.NaN
#film_dose.array[film_dose.array < threshold * np.max(ref_dose)] = np.NaN

# convert distance value from mm to pixels
distTA_pixels = film_dose.dpmm * distTA

# define tolerances and threshold
distance_threshold = distTA_pixels

dose_threshold = doseTA/100 * np.max(ref_dose)
lower_dose_cutoff = threshold * np.max(ref_dose)

# define coordinates
x = ref_dose.shape[0]
y = ref_dose.shape[1]
x_coord =  list(range(0,x))
y_coord =  list(range(0,y))
coords_reference = (x_coord, y_coord)
coords_evaluation = (x_coord, y_coord)


#%%
distance_step_size = distance_threshold / 10    

start = time.time()    
maximum_test_distance = distance_threshold * 1
max_concurrent_calc_points = np.inf
num_threads = 1

if __name__ == '__main__':
    gamma = calc_gamma(coords_reference, ref_dose.array,coords_evaluation, film_dose.array, distance_threshold, dose_threshold, lower_dose_cutoff=lower_dose_cutoff, distance_step_size=distance_step_size, maximum_test_distance=maximum_test_distance, max_concurrent_calc_points=max_concurrent_calc_points, num_threads=num_threads)

end = time.time()
print(end - start)

GammaMap = imageRGB.ArrayImage(gamma, dpi=film_dose.dpi)
GammaMap.plotCB(cmap='bwr', clim=[0,2])

      
fail = np.zeros(GammaMap.shape)
fail[(GammaMap.array > 1.0)] = 1
GammaMap.fail = imageRGB.ArrayImage(fail, dpi=film_dose.dpi)

passed = np.zeros(GammaMap.shape)
passed[(GammaMap.array <= 1.0)] = 1
GammaMap.passed = imageRGB.ArrayImage(passed, dpi=film_dose.dpi)

GammaMap.npassed = sum(sum(passed == 1))
GammaMap.nfail = sum(sum(fail == 1))
GammaMap.npixel = GammaMap.npassed + GammaMap.nfail
GammaMap.passRate = GammaMap.npassed / GammaMap.npixel * 100

print(GammaMap.passRate)




#%%
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from npgamma import calc_gamma

with open('data_unequal_grid.yml', 'r') as file:
    data = yaml.load(file)
    
x_reference = data['x_mephisto']
y_reference = data['d_mephisto']
dose_reference = data['mephisto_dose_grid']

x_evaluation = data['x_monaco']
y_evaluation = data['d_monaco']
dose_evaluation = data['monaco_dose_grid']

coords_reference = (y_reference, x_reference)
coords_evaluation = (y_evaluation, x_evaluation)

distance_threshold = 2
distance_step_size = distance_threshold / 10

dose_threshold = 0.02 * np.max(dose_evaluation)

lower_dose_cutoff = np.max(dose_evaluation) * 0.2

maximum_test_distance = distance_threshold * 2
max_concurrent_calc_points = 30000000
num_threads = 4

if __name__ == '__main__':
    gamma = calc_gamma(coords_reference, dose_reference,coords_evaluation, dose_evaluation, distance_threshold, dose_threshold, lower_dose_cutoff=lower_dose_cutoff, distance_step_size=distance_step_size, maximum_test_distance=maximum_test_distance, max_concurrent_calc_points=max_concurrent_calc_points, num_threads=num_threads)

valid_gamma = gamma[~np.isnan(gamma)]
valid_gamma[valid_gamma > 2] = 2

plt.hist(valid_gamma, 30);
plt.xlim([0,2])

np.sum(valid_gamma <= 1) / len(valid_gamma)

dx = x_evaluation[1] - x_evaluation[0]
x_pcolor = np.arange(x_evaluation[0]-dx/2, x_evaluation[-1] + dx, dx)

dy = y_evaluation[1] - y_evaluation[0]
y_pcolor = np.arange(y_evaluation[0]-dy/2, y_evaluation[-1] + dy, dy)

cut_off_gamma = np.ma.array (gamma, mask=np.isnan(gamma))
cmap = cm.viridis
cmap.set_bad('white',1.)

plt.pcolormesh(
    x_pcolor, y_pcolor, cut_off_gamma, cmap=cmap, vmin=0, vmax=2)

plt.gca().invert_yaxis()
plt.colorbar()

plt.xlabel('x (mm)')
plt.ylabel('depth (mm)')


#%%
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt

from npgamma import calc_gamma

dcm_ref = dicom.read_file("data_reference.dcm")
dcm_evl = dicom.read_file("data_evaluation.dcm")

# The x, y, and z defined here have not been sufficiently verified
# They do not necessarily match either what is within Dicom nor what is within your
# TPS. Please verify these and see if they are what you expect them to be.

# If these functions are incorrect or there is a better choice of dimension definitions 
# please contact me by creating an issue within the github repository:
#   https://github.com/SimonBiggs/npgamma/issues

# If you are able to validate these functions please contact me in the same way.


def load_dose_from_dicom(dcm):
    """Imports the dose in matplotlib format, with the following index mapping:
        i = y
        j = x
        k = z
    
    Therefore when using this function to have the coords match the same order,
    ie. coords_reference = (y, x, z)
    """
    pixels = np.transpose(
        dcm.pixel_array, (1, 2, 0))
    dose = pixels * dcm.DoseGridScaling

    return dose


def load_xyz_from_dicom(dcm):
    """Although this coordinate pull from Dicom works in the scenarios tested
    this is not an official x, y, z pull. It needs further confirmation.
    """
    resolution = np.array(
        dcm.PixelSpacing).astype(float)
    # Does the first index match x? 
    # Haven't tested with differing grid sizes in x and y directions.
    dx = resolution[0]

    # The use of dcm.Columns here is under question
    x = (
        dcm.ImagePositionPatient[0] +
        np.arange(0, dcm.Columns * dx, dx))

    # Does the second index match y? 
    # Haven't tested with differing grid sizes in x and y directions.
    dy = resolution[1]
    
    # The use of dcm.Rows here is under question
    y = (
        dcm.ImagePositionPatient[1] +
        np.arange(0, dcm.Rows * dy, dy))
    
    # Is this correct?
    z = (
        np.array(dcm.GridFrameOffsetVector) +
        dcm.ImagePositionPatient[2])

    return x, y, z


dose_reference = load_dose_from_dicom(dcm_ref)
dose_evaluation = load_dose_from_dicom(dcm_evl)

x_reference, y_reference, z_reference = load_xyz_from_dicom(dcm_ref)
x_evaluation, y_evaluation, z_evaluation = load_xyz_from_dicom(dcm_evl)

# Input coordinates need to match the same order as the dose grid in 
# index reference order.

coords_reference = (  y_reference, x_reference, z_reference)

coords_evaluation = ( y_evaluation, x_evaluation, z_evaluation)

distance_threshold = 3
distance_step_size = distance_threshold / 10

dose_threshold = 0.03 * np.max(dose_reference)
lower_dose_cutoff = np.max(dose_reference) * 0.2
maximum_test_distance = distance_threshold * 2
max_concurrent_calc_points = 30000000
num_threads = 4

if __name__ == '__main__':
    gamma = calc_gamma(coords_reference, dose_reference,coords_evaluation, dose_evaluation, distance_threshold, dose_threshold, lower_dose_cutoff=lower_dose_cutoff, distance_step_size=distance_step_size, maximum_test_distance=maximum_test_distance, max_concurrent_calc_points=max_concurrent_calc_points, num_threads=num_threads)

#%%
valid_gamma = gamma[~np.isnan(gamma)]
valid_gamma[valid_gamma > 2] = 2

plt.hist(valid_gamma, 30);
plt.xlim([0,2])

np.sum(valid_gamma <= 1) / len(valid_gamma)

relevant_slice = (
    np.max(dose_evaluation, axis=(0, 1)) > 
    lower_dose_cutoff)
slice_start = np.max([
        np.where(relevant_slice)[0][0], 
        0])
slice_end = np.min([
        np.where(relevant_slice)[0][-1], 
        len(z_evaluation)])
    
max_ref_dose = np.max(dose_reference)

cut_off_gamma = gamma.copy()
greater_than_2_ref = (cut_off_gamma > 2) & ~np.isnan(cut_off_gamma)
cut_off_gamma[greater_than_2_ref] = 2

for z_i in z_evaluation[slice_start:slice_end:5]:
    i = np.where(z_i == z_evaluation)[0][0]
    j = np.where(z_i == z_reference)[0][0]
    print("======================================================================")
    print("Slice = {0}".format(z_i))  
   
    plt.contourf(
        x_evaluation, y_evaluation, dose_evaluation[:, :, j], 30, 
        vmin=0, vmax=max_ref_dose, cmap=plt.get_cmap('gist_heat'))
    plt.title("Evaluation")
    plt.colorbar()
    plt.show()
    
    plt.contourf(
        x_reference, y_reference, dose_reference[:, :, j], 30, 
        vmin=0, vmax=max_ref_dose, cmap=plt.get_cmap('gist_heat'))
    plt.title("Reference")  
    plt.colorbar()
    plt.show()
    
    plt.contourf(
        x_evaluation, y_evaluation, cut_off_gamma[:, :, i], 30, 
        vmin=0, vmax=2, cmap=plt.get_cmap('bwr'))
    plt.title("Gamma")    
    plt.colorbar()  
    plt.show()
    
    print("\n")